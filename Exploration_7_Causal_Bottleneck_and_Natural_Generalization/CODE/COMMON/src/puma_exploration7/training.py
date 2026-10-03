from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .checkpoint import load_exploration6_head


def _adamw(params, lr, weight_decay, device):
    kw = dict(lr=lr, weight_decay=weight_decay)
    # Fused AdamW is mathematically the same optimizer and removes Python kernel-launch overhead.
    if torch.device(device).type == 'cuda':
        try:
            return torch.optim.AdamW(params, fused=True, **kw)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(params, **kw)


def _as_device_tensor(x, dtype, device):
    a = np.array(x, copy=True, order='C')
    return torch.from_numpy(a).to(device=device, dtype=dtype, non_blocking=True)


def train_crt_output(
    features,
    tier,
    labels,
    base_checkpoint,
    out_path=None,
    seed=17,
    epochs=10,
    lr=1e-3,
    weight_decay=1e-2,
    batch_size=64,
    device='cpu',
):
    """Natural-risk cRT output relearning with the Exploration-6 feature transforms frozen.

    The full cached dataset is copied to the target device once; epochs then index those
    resident tensors directly. This preserves the exact fixed batch size/optimizer recipe
    while avoiding DataLoader and repeated host-to-device overhead.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    base, _, _ = load_exploration6_head(base_checkpoint, device)
    base.freeze_feature_transform()
    base.reset_classifier_outputs()
    base.train()
    trainable = list(base.classifier_parameters())
    opt = _adamw(trainable, lr, weight_decay, device)
    loss_fn = torch.nn.CrossEntropyLoss()
    X = _as_device_tensor(features, torch.float32, device)
    T = _as_device_tensor(tier, torch.float32, device)
    Y = _as_device_tensor(labels, torch.long, device)
    if X.ndim != 2 or X.shape[1] != 1536 or T.shape != (len(X), 16) or Y.shape != (len(X),):
        raise ValueError('training array shape mismatch')
    n = len(Y)
    bs = int(batch_size)
    hist = []
    gen = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(1, int(epochs) + 1):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for start in range(0, n, bs):
            ix = perm[start : start + bs].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            z = base(X[ix], T[ix])
            l = loss_fn(z, Y[ix])
            l.backward()
            opt.step()
            total += float(l.detach()) * len(ix)
        hist.append({'epoch': ep, 'loss': total / max(n, 1)})
    base.eval()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model': base.state_dict(),
                'base_checkpoint': str(base_checkpoint),
                'seed': seed,
                'epochs': int(epochs),
            },
            out_path,
        )
    return base, hist


def infer_a5(model, features, tier, batch_size: int | None = None, device='cpu'):
    features = np.asarray(features, np.float32)
    tier = np.asarray(tier, np.float32)
    n = len(features)
    if n == 0:
        return np.empty((0, 10), np.float32)
    bs = (
        (min(max(1024, n), 16384) if str(device).startswith('cuda') else min(max(512, n), 8192))
        if not batch_size or batch_size <= 0
        else int(batch_size)
    )
    out = np.empty((n, 10), np.float32)
    model.eval()
    dev = torch.device(device)
    with torch.inference_mode():
        for i in range(0, n, bs):
            j = min(i + bs, n)
            h = _as_device_tensor(features[i:j], torch.float32, dev)
            b = _as_device_tensor(tier[i:j], torch.float32, dev)
            out[i:j] = model(h, b).float().cpu().numpy()
    return out
