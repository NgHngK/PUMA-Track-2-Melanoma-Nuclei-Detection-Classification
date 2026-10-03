from __future__ import annotations
from collections import OrderedDict
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .augmentation import centered_crop_with_token_coordinate, image_to_tensor
from .models.uni2 import UNI2Backbone, checkpoint_sha256
from .models.pooling import gaussian_pool
from .manifest import read_manifest
from .cache_integrity import write_cache_sidecar


class _DS(Dataset):
    def __init__(self, df, fov, cache_rois=4):
        self.df = df.reset_index(drop=True)
        self.fov = int(fov)
        self.cache_rois = int(cache_rois)
        self.cache = OrderedDict()
        # Stable ROI grouping reduces TIFF/FUSE I/O while restoring outputs to manifest order.
        self.order = np.argsort(self.df['image'].astype(str).to_numpy(), kind='stable')

    def _image(self, path):
        if path not in self.cache:
            with Image.open(path) as im:
                self.cache[path] = im.convert('RGB').copy()
            while len(self.cache) > self.cache_rois:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        return self.cache[path]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, j):
        i = int(self.order[j])
        r = self.df.iloc[i]
        im = self._image(str(r.image))
        crop, u, v = centered_crop_with_token_coordinate(im, float(r.x), float(r.y), self.fov)
        return image_to_tensor(crop), np.float32(u), np.float32(v), i


def _auto_workers(requested):
    if requested is not None and int(requested) >= 0:
        return int(requested)
    return min(4, max(0, (os.cpu_count() or 2) // 2))


def _auto_batch(device, requested):
    if requested is not None and int(requested) > 0:
        return int(requested)
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        return 2
    gib = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
    # Exact-FP32 representation contract. L4/24GB => 4; A100/40GB+ => 8.
    return 8 if gib >= 36 else (4 if gib >= 20 else 2)


class FrozenUNI2Extractor:
    """Load the 2.7GB UNI2-h backbone once and reuse it for every Exploration-7 cohort/FOV.

    Reusing the resident encoder is a runtime optimization only: crops, FP32 UNI2 forward,
    Gaussian pooling, LayerNorm and output ordering are unchanged.
    """

    def __init__(self, weights, device='cuda', batch_size=0, num_workers=-1, weights_sha256=None):
        self.weights = str(weights)
        self.device = torch.device(
            device if device != 'cuda' or torch.cuda.is_available() else 'cpu'
        )
        self.workers = _auto_workers(num_workers)
        self.batch_size = _auto_batch(self.device, batch_size)
        self.weights_sha256 = str(weights_sha256) if weights_sha256 else checkpoint_sha256(weights)
        self.encoder = UNI2Backbone(weights).to(self.device).eval()

    def extract(self, manifest, out, fov=96, sigma=1.5):
        df = read_manifest(manifest, require_images=True)
        ds = _DS(df, fov)
        kw = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=(self.device.type == 'cuda'),
        )
        if self.workers > 0:
            kw.update(persistent_workers=True, prefetch_factor=2)
        loader = DataLoader(ds, **kw)
        arr = np.empty((len(df), 1536), np.float32)
        with torch.inference_mode():
            for x, u, v, idx in loader:
                x = x.to(self.device, non_blocking=True)
                u = u.to(self.device, non_blocking=True)
                v = v.to(self.device, non_blocking=True)
                tokens = self.encoder.forward_tokens(x)
                h = gaussian_pool(tokens, u, v, sigma)
                h = torch.nn.functional.layer_norm(h.float(), (1536,))
                arr[idx.numpy()] = h.cpu().numpy()
        np.save(out, arr)
        write_cache_sidecar(
            out,
            manifest,
            df,
            'gaussian_features',
            weights_sha256=self.weights_sha256,
            fov=int(fov),
            sigma=float(sigma),
            shape=list(arr.shape),
            extraction_batch_size=self.batch_size,
            num_workers=self.workers,
        )
        return arr

    def close(self):
        self.encoder.cpu()
        del self.encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_gaussian_features(
    manifest,
    weights,
    out,
    fov=96,
    sigma=1.5,
    batch_size=0,
    device='cuda',
    num_workers=-1,
    weights_sha256=None,
):
    extractor = FrozenUNI2Extractor(weights, device, batch_size, num_workers, weights_sha256)
    try:
        return extractor.extract(manifest, out, fov, sigma)
    finally:
        extractor.close()
