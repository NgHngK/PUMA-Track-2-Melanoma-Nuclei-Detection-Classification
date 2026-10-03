from __future__ import annotations
import torch


def parameter_audit(model, before=None):
    named = dict(model.named_parameters())
    total = sum(p.numel() for p in named.values())
    train = {n: p for n, p in named.items() if p.requires_grad}
    frozen = {n: p for n, p in named.items() if not p.requires_grad}
    result = {
        'total_parameters': total,
        'trainable_parameters': sum(p.numel() for p in train.values()),
        'frozen_parameters': sum(p.numel() for p in frozen.values()),
        'trainable_names': list(train),
        'gradient_norms': {
            n: (float(p.grad.norm().detach().cpu()) if p.grad is not None else None)
            for n, p in train.items()
        },
    }
    if before is not None:
        result['parameter_delta_norms'] = {
            n: float((p.detach().cpu() - before[n]).norm()) for n, p in named.items() if n in before
        }
    return result


def assert_frozen_unchanged(model, before):
    for n, p in model.named_parameters():
        if not p.requires_grad and n in before and not torch.equal(p.detach().cpu(), before[n]):
            raise RuntimeError(f'frozen parameter changed: {n}')
