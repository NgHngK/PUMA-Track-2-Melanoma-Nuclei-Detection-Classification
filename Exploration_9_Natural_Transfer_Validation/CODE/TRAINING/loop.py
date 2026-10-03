from __future__ import annotations

import gc
import time
import numpy as np
import torch
from torch.nn import functional as F

from CODE.DATA.sampler import epoch_sample_indices
from CODE.COMMON.constants import (
    BATCH_SIZE,
    TRAIN_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRAD_CLIP,
    CLASSES,
)
from CODE.COMMON.validation import require_finite
from CODE.EVALUATION.epoch_metrics import classification_epoch_metrics, prepare_epoch_metric_context
from CODE.TRAINING.checkpointing import save_training_checkpoint, load_training_checkpoint
from CODE.TRAINING.histories import save_fit_history


def _tensor_on_device(x, dtype, device):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _forward_logits(model, hl, b, hw=None, *, fast=False):
    if fast and hasattr(model, "forward_logits_fast"):
        return (
            model.forward_logits_fast(hl, b) if hw is None else model.forward_logits_fast(hl, hw, b)
        )
    return (model(hl, b) if hw is None else model(hl, hw, b)).logits


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        "cuda" in text and "out of memory" in text
    )


def _clear_cuda_after_oom():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _EvalBatchController:
    """Choose the largest practical FP32 diagnostic batch and reuse it for the fit."""

    def __init__(self, device: str, *, fast: bool = False):
        self.device = str(device)
        self.fast = bool(fast)
        self.batch_size: int | None = None

    @torch.inference_mode()
    def predict(self, model, hl, b, hw=None):
        model.eval()
        n = len(hl)
        if n == 0:
            return np.empty((0, 10), dtype=np.float32)

        if self.batch_size is None:
            candidate = n
            first = None
            while candidate >= 1:
                try:
                    first = _forward_logits(
                        model,
                        hl[:candidate],
                        b[:candidate],
                        None if hw is None else hw[:candidate],
                        fast=self.fast,
                    )
                    if self.device.startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize(torch.device(self.device))
                    self.batch_size = int(candidate)
                    break
                except RuntimeError as exc:
                    if not _is_cuda_oom(exc):
                        raise
                    _clear_cuda_after_oom()
                    candidate //= 2
            if self.batch_size is None:
                raise RuntimeError("Evaluation inference OOM even at batch size 1")
            print(f"[EVAL-AUTO] diagnostic batch={self.batch_size}", flush=True)
            out = [first.float().cpu()]
            start = self.batch_size
        else:
            out = []
            start = 0

        bs = int(self.batch_size)
        while start < n:
            stop = min(start + bs, n)
            try:
                logits = _forward_logits(
                    model,
                    hl[start:stop],
                    b[start:stop],
                    None if hw is None else hw[start:stop],
                    fast=self.fast,
                )
                out.append(logits.float().cpu())
                start = stop
            except RuntimeError as exc:
                if not _is_cuda_oom(exc):
                    raise
                _clear_cuda_after_oom()
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                self.batch_size = bs
                print(f"[EVAL-OOM] reducing diagnostic batch to {bs}", flush=True)
        return torch.cat(out, dim=0).numpy()


def _ce_from_logits(logits, labels):
    if len(labels) == 0:
        return 0.0
    return float(
        F.cross_entropy(
            torch.as_tensor(logits, dtype=torch.float32),
            torch.as_tensor(np.asarray(labels), dtype=torch.long),
        ).item()
    )


def _softmax_np(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_feature_head(
    model,
    h_local,
    tier_a,
    labels,
    *,
    h_wide=None,
    seed=17,
    fold=0,
    epochs=TRAIN_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    device="cpu",
    train_rois=None,
    train_coords=None,
    val_h_local=None,
    val_tier_a=None,
    val_labels=None,
    val_h_wide=None,
    val_rois=None,
    val_coords=None,
    run_tag=None,
    resume_dir=None,
    resume_metadata=None,
    execution_mode="fast",
    return_last_val_logits=False,
):
    """Train one locked E9 head; checkpoint after every completed epoch.

    ``fast`` preserves the exact scientific recipe but removes avoidable CPU↔GPU
    synchronization from per-mini-batch diagnostics. ``strict`` retains the older
    per-step finite checks for debugging/reproduction diagnostics.
    """
    execution_mode = str(execution_mode).lower()
    if execution_mode not in {"fast", "strict"}:
        raise ValueError(f"Unknown execution_mode={execution_mode!r}")
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8
    )
    labels_np = np.asarray(labels, dtype=np.int64)
    hl_all = _tensor_on_device(h_local, torch.float32, device)
    b_all = _tensor_on_device(tier_a, torch.float32, device)
    y_all = _tensor_on_device(labels_np, torch.long, device)
    hw_all = None if h_wide is None else _tensor_on_device(h_wide, torch.float32, device)

    has_epoch_metrics = train_rois is not None and train_coords is not None
    train_metric_context = None
    if has_epoch_metrics:
        train_rois = np.asarray(train_rois)
        train_coords = np.asarray(train_coords, dtype=np.float64)
        if not (len(train_rois) == len(train_coords) == len(labels_np)):
            raise ValueError("Train epoch metric metadata length mismatch")
        train_metric_context = prepare_epoch_metric_context(labels_np, train_rois, train_coords)

    has_val = val_h_local is not None
    val_metric_context = None
    if has_val:
        if any(x is None for x in (val_tier_a, val_labels, val_rois, val_coords)):
            raise ValueError("Incomplete held-fold diagnostic inputs")
        vhl = _tensor_on_device(val_h_local, torch.float32, device)
        vb = _tensor_on_device(val_tier_a, torch.float32, device)
        vhw = None if val_h_wide is None else _tensor_on_device(val_h_wide, torch.float32, device)
        val_labels = np.asarray(val_labels, dtype=np.int64)
        val_rois = np.asarray(val_rois)
        val_coords = np.asarray(val_coords, dtype=np.float64)
        if not (len(vhl) == len(vb) == len(val_labels) == len(val_rois) == len(val_coords)):
            raise ValueError("Held-fold diagnostic input length mismatch")
        val_metric_context = prepare_epoch_metric_context(val_labels, val_rois, val_coords)

    history = []
    start_epoch = 0
    checkpoint_path = None
    if resume_dir is not None:
        from pathlib import Path

        resume_dir = Path(resume_dir)
        resume_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = resume_dir / "latest.pt"
        if checkpoint_path.is_file():
            if resume_metadata is None:
                raise ValueError("resume_metadata is required when a training checkpoint exists")
            start_epoch, history = load_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=opt,
                expected_metadata=resume_metadata,
                device=device,
            )
            save_fit_history(resume_dir, history)
            if start_epoch < epochs:
                print(
                    f"[RESUME] {run_tag or 'fit'}: {start_epoch}/{epochs} epochs complete -> starting epoch {start_epoch + 1}",
                    flush=True,
                )
            else:
                print(
                    f"[RESUME] {run_tag or 'fit'}: {start_epoch}/{epochs} epochs already complete",
                    flush=True,
                )

    eval_controller = _EvalBatchController(device, fast=(execution_mode == "fast"))
    last_val_logits = None

    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        model.train()
        sampled = epoch_sample_indices(labels_np, seed, fold, epoch)
        # FAST: one tiny H2D transfer for the whole epoch instead of one transfer
        # for every 64-sample mini-batch. The ordering is unchanged.
        sampled_device = torch.as_tensor(sampled, dtype=torch.long, device=device)
        n_steps = (len(sampled) + batch_size - 1) // batch_size
        print(
            f"[EPOCH-START] {run_tag or 'fit'} epoch={epoch + 1}/{epochs} "
            f"steps={n_steps} mode={execution_mode}",
            flush=True,
        )

        if execution_mode == "fast":
            loss_sum = torch.zeros((), dtype=torch.float32, device=device)
            grad_sum = torch.zeros((), dtype=torch.float32, device=device)
        else:
            losses = []
            grad_norms = []

        progress_every = max(1, n_steps // 4)
        step_count = 0
        for start in range(0, len(sampled), batch_size):
            ix = sampled_device[start : start + batch_size]
            hl = hl_all.index_select(0, ix)
            b = b_all.index_select(0, ix)
            y = y_all.index_select(0, ix)
            opt.zero_grad(set_to_none=True)
            out = _forward_logits(
                model,
                hl,
                b,
                None if hw_all is None else hw_all.index_select(0, ix),
                fast=(execution_mode == "fast"),
            )
            loss = F.cross_entropy(out, y)
            if execution_mode == "strict":
                require_finite(loss, "loss")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if execution_mode == "strict":
                require_finite(gn, "grad_norm")
            opt.step()

            if execution_mode == "fast":
                # Keep diagnostics on GPU. Calling float(tensor) here would force a
                # CUDA synchronization on every optimizer step and dominate this
                # very small classifier.
                loss_sum.add_(loss.detach().float())
                grad_sum.add_(gn.detach().float())
            else:
                losses.append(float(loss.detach()))
                grad_norms.append(float(gn))

            step_count += 1
            if step_count % progress_every == 0 or step_count == n_steps:
                print(
                    f"[STEP] {run_tag or 'fit'} epoch={epoch + 1}/{epochs} "
                    f"{step_count}/{n_steps}",
                    flush=True,
                )

        if execution_mode == "fast":
            # One synchronization per epoch instead of ~1,200. A non-finite epoch
            # is rejected before checkpointing, so resume restarts from the last
            # known-good completed epoch.
            require_finite(loss_sum, "epoch_loss_sum")
            require_finite(grad_sum, "epoch_grad_norm_sum")
            sampled_loss = float((loss_sum / max(1, step_count)).item())
            mean_grad_norm = float((grad_sum / max(1, step_count)).item())
        else:
            sampled_loss = float(np.mean(losses))
            mean_grad_norm = float(np.mean(grad_norms))

        train_seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch + 1,
            "loss": sampled_loss,
            "sampled_train_loss": sampled_loss,
            "grad_norm": mean_grad_norm,
            "lr": float(opt.param_groups[0]["lr"]),
            "train_seconds": float(train_seconds),
            "execution_mode": execution_mode,
        }
        if has_epoch_metrics:
            train_logits = eval_controller.predict(model, hl_all, b_all, hw_all)
            train_probs = _softmax_np(train_logits)
            row["train"] = classification_epoch_metrics(
                y_true=labels_np,
                probs=train_probs,
                rois=train_rois,
                coords=train_coords,
                loss=_ce_from_logits(train_logits, labels_np),
                context=train_metric_context,
            )
        if has_val:
            val_logits = eval_controller.predict(model, vhl, vb, vhw)
            last_val_logits = val_logits
            val_probs = _softmax_np(val_logits)
            row["val"] = classification_epoch_metrics(
                y_true=val_labels,
                probs=val_probs,
                rois=val_rois,
                coords=val_coords,
                loss=_ce_from_logits(val_logits, val_labels),
                context=val_metric_context,
            )

        if run_tag and has_epoch_metrics and has_val:
            tr = row["train"]
            va = row["val"]
            print(
                f"\n[EPOCH {epoch + 1:02d}/{epochs}] {run_tag} | "
                f"sampled_loss={row['sampled_train_loss']:.4f} | grad_norm={row['grad_norm']:.4f} | lr={row['lr']:.6g}",
                flush=True,
            )
            print(
                f"  TRAIN | CE={tr['loss']:.4f} | ROI-P/R/F1={tr['roi_precision']:.4f}/{tr['roi_recall']:.4f}/{tr['roi_f1']:.4f} | "
                f"Macro-P/R/F1={tr['precision']:.4f}/{tr['recall']:.4f}/{tr['f1']:.4f} | Acc={tr['accuracy']:.4f} | n={tr['n']} | ROIs={tr['roi_count']}",
                flush=True,
            )
            print(
                f"  VAL   | CE={va['loss']:.4f} | ROI-P/R/F1={va['roi_precision']:.4f}/{va['roi_recall']:.4f}/{va['roi_f1']:.4f} | "
                f"Macro-P/R/F1={va['precision']:.4f}/{va['recall']:.4f}/{va['f1']:.4f} | Acc={va['accuracy']:.4f} | n={va['n']} | ROIs={va['roi_count']}",
                flush=True,
            )
            print(
                "  CLASS             | TRAIN: ROI-F1  F1     P      R    SUP  +ROI-F1 | "
                "VAL: ROI-F1  F1     P      R    SUP  +ROI-F1",
                flush=True,
            )
            for cls in CLASSES:
                tc = tr["per_class"][cls]
                vc = va["per_class"][cls]
                print(
                    f"  {cls:<17} | "
                    f"{tc['roi_f1']:.4f}  {tc['f1']:.4f}  {tc['precision']:.4f}  {tc['recall']:.4f}  {tc['support']:5d}  {tc['positive_roi_f1']:.4f} | "
                    f"{vc['roi_f1']:.4f}  {vc['f1']:.4f}  {vc['precision']:.4f}  {vc['recall']:.4f}  {vc['support']:5d}  {vc['positive_roi_f1']:.4f}",
                    flush=True,
                )

        history.append(row)
        if checkpoint_path is not None:
            if resume_metadata is None:
                raise ValueError("resume_metadata is required for resumable training")
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=opt,
                completed_epochs=epoch + 1,
                history=history,
                metadata=resume_metadata,
            )
            print(f"[CHECKPOINT] {run_tag or 'fit'}: saved epoch {epoch + 1}/{epochs}", flush=True)
    if return_last_val_logits:
        return model, history, last_val_logits
    return model, history


@torch.inference_mode()
def predict_feature_head(
    model, h_local, tier_a, *, h_wide=None, batch_size="auto", device="cpu", execution_mode="fast"
):
    model = model.to(device)
    model.eval()
    hl_all = _tensor_on_device(h_local, torch.float32, device)
    b_all = _tensor_on_device(tier_a, torch.float32, device)
    hw_all = None if h_wide is None else _tensor_on_device(h_wide, torch.float32, device)
    if batch_size == "auto":
        return _EvalBatchController(device, fast=(str(execution_mode).lower() == "fast")).predict(
            model, hl_all, b_all, hw_all
        )

    logits = []
    bs = int(batch_size)
    for start in range(0, len(h_local), bs):
        sl = slice(start, start + bs)
        out = _forward_logits(
            model,
            hl_all[sl],
            b_all[sl],
            None if hw_all is None else hw_all[sl],
            fast=(str(execution_mode).lower() == "fast"),
        )
        logits.append(out.float().cpu().numpy())
    return np.concatenate(logits, axis=0)
