"""Extract UNI2-h, Tier-A and NiTo features at nucleus centroids."""

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common.resources import cpu_workers, gpu_batch_size

from .uni2 import load_uni2, crop_at, pool, MEAN, STD
from .nito import tissue20
from .tier import _roi_maps, point_features
from .data import read_rgb_image
from stage1.annotations import normalized_stem


class Extractor:
    def __init__(
        self, uni2_path, nito_dir, device="cuda", batch_size=0, probability_dir=None, workers=0
    ):
        if probability_dir is None:
            raise ValueError("probability_dir is required; prepare verified NiTo maps first")
        if batch_size < 0 or workers < 0:
            raise ValueError("batch_size and workers must be nonnegative")
        self.device = torch.device(device)
        self.model = load_uni2(Path(uni2_path), self.device)
        self.probability_dir = Path(probability_dir)
        self.mean = MEAN.to(self.device)
        self.std = STD.to(self.device)
        self.workers = workers or cpu_workers()
        self.pool = ThreadPoolExecutor(max_workers=self.workers)
        self.prefetch = ThreadPoolExecutor(max_workers=1)
        self.amp_dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

        self.batch_size = batch_size or self._batch_size()
        print(f"UNI2 batch size: {self.batch_size} | device: {self.device}", flush=True)

    @torch.inference_mode()
    def _batch_size(self):
        def run(size):
            sample = torch.zeros(size, 3, 224, 224, device=self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.device.type == "cuda",
            ):
                tokens = self.model.forward_features(sample)

            del sample, tokens

        return gpu_batch_size(self.device, run, limit=2048)

    @staticmethod
    def _crop(item):
        image, x, y, fov = item
        crop, px, py = crop_at(image, x, y, fov)
        resized = np.asarray(crop.resize((224, 224), Image.Resampling.BICUBIC))
        return resized, px, py

    @torch.inference_mode()
    def extract(self, image_path, xy):
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        n = len(xy)

        if not n:
            return {
                "xy": xy,
                "tier_raw": np.empty((0, 16), np.float32),
                "h96": np.empty((0, 1536), np.float32),
                "h192": np.empty((0, 1536), np.float32),
                "tissue": np.empty((0, 20), np.float32),
            }

        image = Image.fromarray(read_rgb_image(image_path))
        image.load()
        maps = _roi_maps(np.asarray(image))
        tier = np.stack([point_features(maps, x, y) for x, y in xy]).astype(np.float32)
        result = {"xy": xy, "tier_raw": tier}

        for fov in (96, 192):
            features = []
            batches = [xy[left : left + self.batch_size] for left in range(0, n, self.batch_size)]

            def prepare_batch(coords):
                rows = self.pool.map(self._crop, ((image, x, y, fov) for x, y in coords))
                crops, px, py = zip(*rows)
                return np.stack(crops), px, py

            next_batch = self.prefetch.submit(prepare_batch, batches[0])

            for i in range(len(batches)):
                batch, px, py = next_batch.result()

                if i + 1 < len(batches):
                    next_batch = self.prefetch.submit(prepare_batch, batches[i + 1])

                tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)
                if self.device.type == "cuda":
                    tensor = tensor.pin_memory()
                tensor = tensor.to(self.device, dtype=torch.float32, non_blocking=True).div_(255)
                tensor = (tensor - self.mean) / self.std

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.device.type == "cuda",
                ):
                    tokens = self.model.forward_features(tensor)

                pooled = pool(tokens.float(), px, py)
                features.append(pooled.cpu().numpy())

            result[f"h{fov}"] = np.concatenate(features).astype(np.float32)

        with np.load(self.probability_dir / (normalized_stem(image_path) + ".npz")) as archive:
            prob = archive["prob"]
            context = (archive["local"], archive["wide"]) if "local" in archive else None
        if prob.shape != (6, image.height, image.width):
            raise ValueError("NiTo map geometry does not match the source image")

        result["tissue"] = tissue20(prob, xy, context=context)
        image.close()
        return result

    def close(self):
        self.prefetch.shutdown(wait=True)
        self.pool.shutdown(wait=True)
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
