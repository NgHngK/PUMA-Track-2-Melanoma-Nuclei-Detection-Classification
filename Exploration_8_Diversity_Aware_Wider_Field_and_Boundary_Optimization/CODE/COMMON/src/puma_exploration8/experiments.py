from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from torch import nn
from .models.a5 import A5Head, AppearanceHead
from .models.routing import WiderResidualA, WiderInteractionC
from .models.stage2 import Stage2CachedModel
from .normalization import fit_tiera_normalizer
from .metrics import classification_metrics
from .replication import seed_all
from .audit import parameter_audit, assert_frozen_unchanged
from .runtime import can_reside_on_cuda, cuda_memory_stats, release_cuda


@dataclass
class Arrays:
    uids: list[str]
    y: np.ndarray
    groups: np.ndarray
    local: np.ndarray
    bio: np.ndarray
    wider: np.ndarray | None = None


def subset_arrays(a: Arrays, ix) -> Arrays:
    ix = np.asarray(ix)
    return Arrays(
        [a.uids[i] for i in ix],
        a.y[ix],
        a.groups[ix],
        a.local[ix],
        a.bio[ix],
        None if a.wider is None else a.wider[ix],
    )


def inverse_indices(y, draws, seed):
    y = np.asarray(y, int)
    counts = np.bincount(y, minlength=10)
    if (counts == 0).any():
        raise ValueError('all ten classes required for inverse-count exposure')
    rng = np.random.default_rng(seed)
    p = 1 / counts[y]
    p = p / p.sum()
    return rng.choice(len(y), size=int(draws), replace=True, p=p)


def group_aware_indices(y, groups, draws, seed):
    y = np.asarray(y, int)
    groups = np.asarray(groups, str)
    rng = np.random.default_rng(seed)
    by = {}
    for c in range(10):
        gs = np.unique(groups[y == c])
        if len(gs) == 0:
            raise ValueError(f'class {c} absent')
        by[c] = [np.where((y == c) & (groups == g))[0] for g in gs]
    out = np.empty(int(draws), dtype=int)
    for i in range(len(out)):
        c = int(rng.integers(0, 10))
        group_rows = by[c]
        rows = group_rows[int(rng.integers(0, len(group_rows)))]
        out[i] = int(rows[int(rng.integers(0, len(rows)))])
    return out


def make_route(kind: str):
    if kind == 'A':
        return WiderResidualA()
    if kind == 'C':
        return WiderInteractionC(rank=8)
    raise ValueError(kind)


def build_stage2_model(
    seed: int, use_tiera: bool = True, route_kind: str | None = None
) -> Stage2CachedModel:
    """Seed *before* every trainable constructor so matched seeds are actually matched."""
    seed_all(seed)
    model = Stage2CachedModel(use_tiera, None)
    if route_kind:
        model.wider_route = make_route(route_kind)
    model._exploration8_init_seed = int(seed)
    return model


def _forward(model, local, bio, wide):
    if isinstance(model, Stage2CachedModel):
        return model(local, bio, wide)
    if isinstance(model, A5Head):
        return model(local, bio)
    if isinstance(model, AppearanceHead):
        return model(local)
    raise TypeError(type(model))


def _fold_id(uids, seed):
    h = hashlib.sha256('\n'.join(sorted(map(str, uids))).encode()).hexdigest()[:12]
    return f'trainuids_{h}_seed{seed}'


def _nbytes(*arrays) -> int:
    return sum(int(np.asarray(x).nbytes) for x in arrays if x is not None)


def _cpu_tensor(a, dtype, pin: bool):
    t = torch.as_tensor(np.asarray(a), dtype=dtype)
    return t.pin_memory() if pin else t


def train_model(
    model,
    train: Arrays,
    val: Arrays,
    seed=17,
    epochs=10,
    lr=1e-3,
    wd=0.01,
    batch=64,
    draws=None,
    exposure='inverse',
    device='cpu',
    trainable_filter=None,
    track_history=True,
):
    if getattr(model, '_exploration8_init_seed', None) != int(seed):
        raise ValueError(
            'model initialization seed is not bound to this run; use build_stage2_model(seed, ...)'
        )
    if exposure not in {'inverse', 'group'}:
        raise ValueError(exposure)
    seed_all(seed)
    dev = torch.device(device)
    model = model.to(dev)
    if trainable_filter is not None:
        for name, p in model.named_parameters():
            p.requires_grad_(bool(trainable_filter(name)))
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    norm = fit_tiera_normalizer(train.bio, train.uids, _fold_id(train.uids, seed))
    train_bio = norm.transform(train.bio)
    val_bio = norm.transform(val.bio)

    required = _nbytes(
        train.local, train_bio, train.wider, train.y, val.local, val_bio, val.wider, val.y
    )
    resident = dev.type == 'cuda' and can_reside_on_cuda(
        required * 2
    )  # leave working-memory headroom
    pin = dev.type == 'cuda' and not resident

    def pack(a: Arrays, bio):
        items = (
            _cpu_tensor(a.local, torch.float32, pin),
            _cpu_tensor(bio, torch.float32, pin),
            None if a.wider is None else _cpu_tensor(a.wider, torch.float32, pin),
            _cpu_tensor(a.y, torch.long, pin),
        )
        if resident:
            return tuple(None if x is None else x.to(dev, non_blocking=True) for x in items)
        return items

    tl, tb, tw, ty = pack(train, train_bio)
    vl, vb, vw, vy = pack(val, val_bio)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('no trainable params')
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    draws = int(draws or len(train.y))
    hist, last_grad = [], {}
    exposure_counts = np.zeros(10, dtype=np.int64)
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)

    def take(t, idx):
        if t is None:
            return None
        return t[idx] if resident else t[idx.cpu()].to(dev, non_blocking=True)

    def predict_val():
        model.eval()
        preds = []
        with torch.inference_mode():
            for start in range(0, len(val.y), int(batch)):
                sl = slice(start, min(start + int(batch), len(val.y)))
                if resident:
                    zz = _forward(model, vl[sl], vb[sl], None if vw is None else vw[sl])
                else:
                    ll = vl[sl].to(dev, non_blocking=True)
                    bb = vb[sl].to(dev, non_blocking=True)
                    ww = None if vw is None else vw[sl].to(dev, non_blocking=True)
                    zz = _forward(model, ll, bb, ww)
                preds.append(zz.float().cpu())
                if not resident:
                    del ll, bb, ww, zz
        return torch.cat(preds).numpy()

    epochs = int(epochs)
    if epochs < 1:
        raise ValueError('epochs must be >= 1')
    for ep in range(epochs):
        idx_np = (
            inverse_indices(train.y, draws, seed + ep)
            if exposure == 'inverse'
            else group_aware_indices(train.y, train.groups, draws, seed + ep)
        )
        exposure_counts += np.bincount(train.y[idx_np], minlength=10)
        model.train()
        total = 0.0
        for start in range(0, len(idx_np), int(batch)):
            ids = idx_np[start : start + int(batch)]
            idx = torch.as_tensor(ids, dtype=torch.long, device=dev if resident else 'cpu')
            opt.zero_grad(set_to_none=True)
            z = _forward(model, take(tl, idx), take(tb, idx), take(tw, idx))
            target = take(ty, idx)
            loss = nn.functional.cross_entropy(z, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss')
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            if ep == epochs - 1 and start + int(batch) >= len(idx_np):
                last_grad = {
                    n: None if p.grad is None else float(p.grad.detach().norm().cpu())
                    for n, p in model.named_parameters()
                    if p.requires_grad
                }
            opt.step()
            if track_history:
                total += float(loss.detach().cpu()) * len(ids)
            if not resident:
                del z, target, loss
        if track_history:
            vz = predict_val()
            hist.append(
                {
                    'epoch': ep + 1,
                    'objective': total / len(idx_np),
                    'val': classification_metrics(val.y, vz),
                }
            )

    if not track_history:
        vz = predict_val()

    model = model.to('cpu')
    assert_frozen_unchanged(model, before)
    audit = parameter_audit(model, before)
    audit.update(
        last_gradient_norms=last_grad,
        normalizer_fold_id=norm.fold_id,
        feature_storage='gpu_resident' if resident else ('pinned_cpu_streaming' if pin else 'cpu'),
        sampled_class_counts=exposure_counts.tolist(),
        effective_source_prior=(exposure_counts / max(1, exposure_counts.sum())).tolist(),
    )
    if dev.type == 'cuda':
        audit['cuda_memory'] = cuda_memory_stats()
    if (
        isinstance(model, Stage2CachedModel)
        and model.wider_route is not None
        and val.wider is not None
    ):
        with torch.inference_mode():
            ll = torch.as_tensor(val.local[: min(len(val.y), 1024)], dtype=torch.float32)
            bb = torch.as_tensor(val_bio[: len(ll)], dtype=torch.float32)
            ww = torch.as_tensor(val.wider[: len(ll)], dtype=torch.float32)
            base = model.local(ll, bb) if model.use_tiera else model.local(ll)
            delta = model.wider_route(ll, ww)
        b = float(base.abs().mean())
        d = float(delta.abs().mean())
        audit['base_logit_mean_abs'] = b
        audit['residual_logit_mean_abs'] = d
        audit['residual_to_base_ratio'] = d / max(b, 1e-12)
        if hasattr(model.wider_route, 'scale'):
            audit['global_residual_scale'] = float(model.wider_route.scale().detach())
    del opt, params, tl, tb, tw, ty, vl, vb, vw, vy
    if dev.type == 'cuda':
        release_cuda()
    return model, norm, vz, hist, audit


def train_full_model(
    model,
    train: Arrays,
    seed=17,
    epochs=10,
    lr=1e-3,
    wd=0.01,
    batch=64,
    draws=None,
    exposure='inverse',
    device='cpu',
):
    """Fixed-budget full-TRAIN refit after all architecture choices are frozen."""
    if getattr(model, '_exploration8_init_seed', None) != int(seed):
        raise ValueError(
            'model initialization seed is not bound to this run; use build_stage2_model(seed, ...)'
        )
    seed_all(seed)
    dev = torch.device(device)
    model = model.to(dev)
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    norm = fit_tiera_normalizer(train.bio, train.uids, f'FULL_TRAIN_seed{seed}')
    bio = norm.transform(train.bio)
    required = _nbytes(train.local, bio, train.wider, train.y)
    resident = dev.type == 'cuda' and can_reside_on_cuda(required * 2)
    pin = dev.type == 'cuda' and not resident
    tl = _cpu_tensor(train.local, torch.float32, pin)
    tb = _cpu_tensor(bio, torch.float32, pin)
    tw = None if train.wider is None else _cpu_tensor(train.wider, torch.float32, pin)
    ty = _cpu_tensor(train.y, torch.long, pin)
    if resident:
        tl, tb, ty = tl.to(dev), tb.to(dev), ty.to(dev)
        if tw is not None:
            tw = tw.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    draws = int(draws or len(train.y))
    exposure_counts = np.zeros(10, dtype=np.int64)
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)
    for ep in range(int(epochs)):
        idx_np = (
            inverse_indices(train.y, draws, seed + ep)
            if exposure == 'inverse'
            else group_aware_indices(train.y, train.groups, draws, seed + ep)
        )
        exposure_counts += np.bincount(train.y[idx_np], minlength=10)
        model.train()
        for start in range(0, len(idx_np), int(batch)):
            ids = idx_np[start : start + int(batch)]
            if resident:
                ii = torch.as_tensor(ids, dtype=torch.long, device=dev)
                ll, bb, ww, yy = tl[ii], tb[ii], None if tw is None else tw[ii], ty[ii]
            else:
                ii = torch.as_tensor(ids, dtype=torch.long)
                ll = tl[ii].to(dev, non_blocking=True)
                bb = tb[ii].to(dev, non_blocking=True)
                ww = None if tw is None else tw[ii].to(dev, non_blocking=True)
                yy = ty[ii].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(_forward(model, ll, bb, ww), yy)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss')
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            opt.step()
            if not resident:
                del ll, bb, ww, yy, loss
    model = model.to('cpu')
    assert_frozen_unchanged(model, before)
    audit = parameter_audit(model, before)
    audit.update(
        feature_storage='gpu_resident' if resident else ('pinned_cpu_streaming' if pin else 'cpu'),
        sampled_class_counts=exposure_counts.tolist(),
        effective_source_prior=(exposure_counts / max(1, exposure_counts.sum())).tolist(),
        normalizer_fold_id=norm.fold_id,
    )
    if dev.type == 'cuda':
        audit['cuda_memory'] = cuda_memory_stats()
    del opt, params, tl, tb, tw, ty
    if dev.type == 'cuda':
        release_cuda()
    return model, norm, audit
