"""One clear command entrypoint for split audit, preprocessing, training and reports."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .splits import load_splits
from .preprocess import prepare
from .train import fit_one, summarize, diagnostic_loss
from .models.build import ARCHITECTURES, build_model


def parser():
    p = argparse.ArgumentParser(
        prog="puma-stage1", description="Standalone PUMA Stage-1 architecture benchmark"
    )
    sub = p.add_subparsers(dest="command", required=True)
    audit = sub.add_parser(
        "audit-splits", help="Verify the provided 4-fold lists without any raw data"
    )
    audit.add_argument("--plan", choices=("A", "B", "both"), default="both")
    prep = sub.add_parser(
        "prepare", help="TIFF + nuclei GeoJSON -> cached RGB/heatmaps/offset targets"
    )
    prep.add_argument("--images", type=Path, required=True)
    prep.add_argument("--annotations", type=Path, required=True)
    prep.add_argument("--cache", type=Path, required=True)
    prep.add_argument("--plan", choices=("A", "B"), default="B")
    prep.add_argument("--roi-size", type=int, default=1024)
    train = sub.add_parser("train", help="Train one fold of one architecture")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("runs"))
    train.add_argument("--plan", choices=("A", "B"), default="B")
    train.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    train.add_argument("--fold", type=int, choices=range(4), required=True)
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--tile-size", type=int, choices=(256, 512, 1024), default=512)
    train.add_argument("--overlap", type=int, default=64)
    train.add_argument(
        "--micro-batch", type=int, default=0, help="0 = choose largest fitting batch automatically"
    )
    train.add_argument("--effective-batch", type=int, default=16)
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--validate-every", type=int, default=1)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--lr", type=float, default=0.00035)
    train.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train official backbone from random initialization",
    )
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--no-ema", action="store_true")
    train.add_argument("--grad-checkpointing", action="store_true")
    train.add_argument("--device", default="auto")
    train.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Override train-only threshold sweep",
    )
    train.add_argument(
        "--radii", type=float, nargs="+", default=None, help="Override train-only NMS-radius sweep"
    )
    many = sub.add_parser(
        "run-all", help="Train all architecture/fold combinations for one or both plans"
    )
    for argument in train._actions:
        if argument.dest in ("help", "architecture", "fold"):
            continue
        opts = [x for x in argument.option_strings if x.startswith("--")]
        if opts:
            kwargs = {"default": argument.default, "help": argument.help}
            if isinstance(argument, argparse._StoreTrueAction):
                kwargs["action"] = "store_true"
            else:
                if argument.type:
                    kwargs["type"] = argument.type
                if argument.choices:
                    kwargs["choices"] = argument.choices
                if argument.nargs:
                    kwargs["nargs"] = argument.nargs
                if argument.required:
                    kwargs["required"] = True
            many.add_argument(*opts, **kwargs)
    for action in many._actions:
        if action.dest == "plan":
            action.choices = ("A", "B", "both")
            break
    doctor = sub.add_parser(
        "doctor", help="Run real timm model forward/backward checks before full training"
    )
    doctor.add_argument("--device", default="auto")
    doctor.add_argument("--size", type=int, default=128)
    doctor.add_argument("--no-pretrained", action="store_true")
    report = sub.add_parser("compare", help="Summarize completed validation folds as mean ± std")
    report.add_argument("--output", type=Path, default=Path("runs"))
    report.add_argument("--plan", choices=("A", "B", "both"), default="both")
    report.add_argument(
        "--allow-partial", action="store_true", help="Permit comparison when some folds are missing"
    )
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.command == "audit-splits":
        for plan in ("A", "B") if a.plan == "both" else (a.plan,):
            folds = load_splits(plan)
            print(f"Plan {plan}")
            for i, d in folds.items():
                print(f"  fold {i}: train={len(d['train'])} val={len(d['val'])} overlap=0")
        return 0
    if a.command == "prepare":
        meta = prepare(a.images, a.annotations, a.cache, plan=a.plan, size=a.roi_size)
        print(
            json.dumps(
                {
                    "num_rois": len(meta["names"]),
                    "nuclei": meta.get("num_nuclei"),
                    "cache_id": meta["id"],
                },
                indent=2,
            )
        )
        return 0
    if a.command == "doctor":
        import gc
        import torch
        from .models.build import TIMM_MODELS

        device = torch.device(
            "cuda"
            if a.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if a.device == "auto"
            else a.device
        )
        if a.size < 64 or a.size % 32:
            raise ValueError("doctor size must be divisible by 32 and >=64")
        if device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        for arch in ("convnext_tiny", "convnext_base"):
            print(
                f"Checking {TIMM_MODELS[arch]} pretrained={not a.no_pretrained} on {device}...",
                flush=True,
            )
            model = build_model(arch, pretrained=not a.no_pretrained).to(device).train()
            x = torch.rand(1, 3, a.size, a.size, device=device)
            result = model(x)
            expected = {
                "heatmap_logits": (1, 1, a.size // 2, a.size // 2),
                "offset": (1, 2, a.size // 2, a.size // 2),
                "quality_logits": (1, 1, a.size // 2, a.size // 2),
                "log_variance": (1, 1, a.size // 2, a.size // 2),
            }
            for key, shape in expected.items():
                if tuple(result[key].shape) != shape:
                    raise RuntimeError(
                        f"{arch}: {key} has shape {tuple(result[key].shape)}, wanted {shape}"
                    )
            loss = diagnostic_loss(result)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{arch}: Nonfinite detector diagnostic loss")
            loss.backward()
            if not all(
                (
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.requires_grad
                )
            ):
                raise RuntimeError(f"{arch}: Missing or nonfinite detector gradients")
            print(
                f"PASS {arch}: output shape, real detector loss, all gradients finite", flush=True
            )
            del model, x, result, loss
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return 0
    if a.command in ("train", "run-all"):
        kw = dict(
            cache_dir=a.cache,
            output=a.output,
            plan=a.plan,
            pretrained=not a.no_pretrained,
            epochs=a.epochs,
            tile_size=a.tile_size,
            overlap=a.overlap,
            micro_batch=a.micro_batch,
            effective_batch=a.effective_batch,
            workers=a.workers,
            validate_every=a.validate_every,
            seed=a.seed,
            lr=a.lr,
            ema_decay=0.0 if a.no_ema else 0.9995,
            amp=not a.no_amp,
            grad_checkpointing=a.grad_checkpointing,
            device_str=a.device,
            train_thresholds=a.thresholds,
            train_radii=a.radii,
        )
        if a.command == "train":
            print(fit_one(**kw, arch=a.architecture, fold=a.fold))
        else:
            plans = ("A", "B") if a.plan == "both" else (a.plan,)
            for plan in plans:
                for architecture in ARCHITECTURES:
                    for fold in range(4):
                        print(fit_one(**{**kw, "plan": plan}, arch=architecture, fold=fold))
                        import gc, torch

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        return 0
    if a.command == "compare":
        for plan in ("A", "B") if a.plan == "both" else (a.plan,):
            print(f"\nPLAN {plan}")
            rows = summarize(a.output, plan, require_complete=not a.allow_partial)
            for row in rows:
                if isinstance(row["fold"], int):
                    print(
                        f"{row['model']:>14s} fold={row['fold']} F1={row['f1']:.4f} P={row['precision']:.4f} R={row['recall']:.4f}"
                    )
                else:
                    print(
                        f"{row['model']:>14s} {row['metric']}: {row['mean']:.4f} ± {row['std']:.4f}"
                    )
            print("Comparison printed only; no sidecar file is written")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
