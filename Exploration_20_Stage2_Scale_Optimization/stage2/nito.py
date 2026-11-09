"""Frozen NiTo ensemble and 20-dimensional tissue features."""

from pathlib import Path
import gc
import json

import numpy as np
import torch
import torch.nn.functional as f
from scipy.ndimage import gaussian_filter, map_coordinates

from .settings import segformer_config
from common.run_history import file_fingerprint, fingerprint
from .data import read_rgb_image


class Nito:
    def __init__(self, weights_dir, device="cuda"):
        self.weights_dir = Path(weights_dir)
        self.device = torch.device(device)

    def load_group(self, group):
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        classes = 11 if group == "classifier" else 6
        config = SegformerConfig.from_dict(segformer_config(classes))
        models = []

        for fold in range(5):
            path = self.weights_dir / f"{group}{fold}" / "checkpoint_epoch1.pth"
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
            state = {name.removeprefix("module."): value for name, value in state.items()}

            model = SegformerForSemanticSegmentation(config)
            model.load_state_dict(state, strict=True)
            models.append(model.to(self.device).eval().requires_grad_(False))

        return models

    def image_tensor(self, image_path):
        rgb = read_rgb_image(image_path)

        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        image = image.to(self.device, dtype=torch.float32).unsqueeze(0) / 255
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (image - mean) / std

    @torch.inference_mode()
    def predict_group(self, models, image):
        probabilities = None

        for model in models:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                logits = model(pixel_values=image).logits
                logits = f.interpolate(
                    logits, size=image.shape[-2:], mode="bilinear", align_corners=False
                )

            result = logits.float().softmax(dim=1)[0]
            probabilities = result if probabilities is None else probabilities + result

        return probabilities / len(models)

    @staticmethod
    def choose_group(probabilities):
        labels = probabilities.argmax(dim=0)

        if bool((labels == 4).any()):
            return "Primary"

        primary = int(((labels >= 1) & (labels <= 5)).sum())
        metastatic = int((labels >= 6).sum())
        return "Primary" if primary > metastatic else "Metastatic"

    @torch.inference_mode()
    def prepare_maps(self, images, output_dir):
        """Cache ensemble routing and probability maps for repeated experiments."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        images = dict(images)
        route_file = output_dir / "routes.json"

        # Bind every cached prediction and route to image bytes and ensemble weights.
        # A legacy cache without this evidence is recomputed rather than adopted.
        model_source = {
            str(path.relative_to(self.weights_dir)): file_fingerprint(path)
            if path.is_file()
            else None
            for group in ("classifier", "Primary", "Metastatic")
            for fold in range(5)
            for path in [self.weights_dir / f"{group}{fold}" / "checkpoint_epoch1.pth"]
        }
        sources = {
            name: fingerprint(
                {
                    "image": file_fingerprint(path),
                    "models": model_source,
                    "revision": "nito-six-class-context-v2",
                }
            )
            for name, path in images.items()
        }
        metadata_file = output_dir / "manifest.json"
        metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        valid_maps = set()
        for name in images:
            destination = output_dir / f"{name}.npz"
            if metadata.get(name) != sources[name] or not destination.is_file():
                continue
            try:
                with np.load(destination, allow_pickle=False) as archive:
                    prob = archive["prob"]
                    local, wide = tissue_context_maps(prob)
                    complete = (
                        "local" in archive
                        and "wide" in archive
                        and archive["local"].shape == local.shape
                        and archive["wide"].shape == wide.shape
                        and np.isfinite(archive["local"]).all()
                        and np.isfinite(archive["wide"]).all()
                    )
                if not complete:
                    temporary = destination.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, prob=prob, local=local, wide=wide)
                    temporary.replace(destination)
                valid_maps.add(name)
            except (ValueError, KeyError, OSError, EOFError):
                continue
        if len(valid_maps) == len(images):
            print("Reusing verified NiTo maps", flush=True)
            return

        saved_routes = json.loads(route_file.read_text()) if route_file.exists() else {}
        groups = {
            name: group
            for name, group in saved_routes.get("routes", {}).items()
            if saved_routes.get("sources", {}).get(name) == sources.get(name)
            and group in ("Primary", "Metastatic")
        }
        missing_routes = [(name, path) for name, path in images.items() if name not in groups]
        if missing_routes:
            models = self.load_group("classifier")
            for index, (name, image_path) in enumerate(missing_routes, 1):
                image = self.image_tensor(image_path)
                groups[name] = self.choose_group(self.predict_group(models, image))
                print(f"NiTo routing {index}/{len(missing_routes)}: {name}", flush=True)
            del models
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            temporary = route_file.with_suffix(".tmp.json")
            temporary.write_text(json.dumps({"routes": groups, "sources": sources}, indent=2))
            temporary.replace(route_file)

        for group in ("Primary", "Metastatic"):
            selected = [
                (name, path)
                for name, path in images.items()
                if groups[name] == group and name not in valid_maps
            ]
            if not selected:
                continue

            models = self.load_group(group)

            for index, (name, image_path) in enumerate(selected, 1):
                image = self.image_tensor(image_path)
                prob = self.predict_group(models, image).cpu().numpy()
                local, wide = tissue_context_maps(prob)
                destination = output_dir / f"{name}.npz"
                temporary = destination.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, prob=prob, local=local, wide=wide)
                temporary.replace(destination)
                metadata[name] = sources[name]
                temporary_metadata = metadata_file.with_suffix(".tmp.json")
                temporary_metadata.write_text(json.dumps(metadata, indent=2))
                temporary_metadata.replace(metadata_file)
                print(f"NiTo {group} {index}/{len(selected)}: {name}", flush=True)

            del models
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def predict(self, rgb):
        """Run a single ROI independently when no precomputed map is available."""
        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        image = image.to(self.device, dtype=torch.float32).unsqueeze(0) / 255
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image = (image - mean) / std

        models = self.load_group("classifier")
        group = self.choose_group(self.predict_group(models, image))
        del models

        models = self.load_group(group)
        probabilities = self.predict_group(models, image).cpu().numpy()
        del models, image
        return probabilities


def bilinear(prob, xy):
    if len(xy) == 0:
        return np.empty((0, prob.shape[0]), dtype=np.float32)

    coords = np.asarray(xy, dtype=np.float64)[:, [1, 0]].T
    return np.stack(
        [map_coordinates(channel, coords, order=1, mode="nearest") for channel in prob],
        axis=1,
    ).astype(np.float32)


def blurred(prob, sigma):
    shape = prob.shape[-2:]
    norm = gaussian_filter(np.ones(shape, np.float32), sigma=sigma, mode="constant")
    smooth = np.stack(
        [gaussian_filter(p, sigma=sigma, mode="constant") / np.maximum(norm, 1e-06) for p in prob]
    )
    return smooth / np.maximum(smooth.sum(axis=0, keepdims=True), 1e-06)


def tissue_context_maps(prob):
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 3 or prob.shape[0] != 6 or not np.isfinite(prob).all() or np.any(prob < 0):
        raise ValueError("NiTo requires finite nonnegative six-class probability maps")
    _, height, width = prob.shape
    if height < 4 or width < 4 or height % 4 or width % 4:
        raise ValueError("NiTo context map height and width must be positive and divisible by 4")
    stride = 4
    low = prob.reshape(6, height // stride, stride, width // stride, stride).mean(
        axis=(2, 4), dtype=np.float32
    )
    return blurred(low, 24 / stride), blurred(low, 48 / stride)


def tissue20(prob, xy, context=None):
    center = bilinear(prob, xy)
    local_map, wide_map = tissue_context_maps(prob) if context is None else context
    points = (np.asarray(xy, dtype=np.float64) + 0.5) / 4 - 0.5
    local = bilinear(local_map, points)
    wide = bilinear(wide_map, points)
    center /= np.maximum(center.sum(1, keepdims=True), 1e-06)
    local /= np.maximum(local.sum(1, keepdims=True), 1e-06)
    wide /= np.maximum(wide.sum(1, keepdims=True), 1e-06)
    entropy = -(center * np.log(np.maximum(center, 1e-08))).sum(1, keepdims=True) / np.log(6)
    disagreement = 0.5 * np.abs(center - wide).sum(1, keepdims=True)
    return np.concatenate([center, local, wide, entropy, disagreement], axis=1).astype(np.float32)
