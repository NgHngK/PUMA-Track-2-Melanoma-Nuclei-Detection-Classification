"""Spatial-neighbor graph: radius 192, 20 neighbors, Gaussian sigma 64."""

import numpy as np
import torch
import torch.nn.functional as f
from scipy.spatial import cKDTree


def build(xy, rois, radius=192.0, k=20, sigma=64.0):
    xy = np.asarray(xy, dtype=np.float64)
    rois = np.asarray(rois)
    if xy.shape != (len(rois), 2) or not np.isfinite(xy).all():
        raise ValueError("Graph requires finite N x 2 coordinates and N ROI names")
    if (
        not np.isfinite(radius)
        or not np.isfinite(sigma)
        or radius <= 0
        or sigma <= 0
        or not isinstance(k, int)
        or k < 1
    ):
        raise ValueError("Graph radius, sigma and neighbor count must be positive")
    n = len(xy)
    neighbors = np.full((n, k), -1, dtype=np.int64)
    distance = np.zeros((n, k), dtype=np.float32)
    for roi in np.unique(rois):
        indices = np.flatnonzero(rois == roi)
        if len(indices) < 2:
            continue
        points = xy[indices]
        tree = cKDTree(points)
        neighborhoods = tree.query_ball_point(points, r=radius)
        for local_id, nearby in enumerate(neighborhoods):
            nearby = np.asarray([j for j in nearby if j != local_id], dtype=np.int64)
            if not len(nearby):
                continue
            distances = np.linalg.norm(points[nearby] - points[local_id], axis=1)
            order = np.lexsort((indices[nearby], distances))[:k]
            selected = nearby[order]
            count = len(selected)
            neighbors[indices[local_id], :count] = indices[selected]
            distance[indices[local_id], :count] = distances[order]
    weights = np.zeros((n, k), dtype=np.float32)
    valid = neighbors >= 0
    weights[valid] = np.exp(-0.5 * (distance[valid] / sigma) ** 2)
    totals = weights.sum(axis=1, keepdims=True)
    np.divide(weights, totals, out=weights, where=totals > 0)
    return {"neighbors": neighbors, "weights": weights, "distance": distance}


def gpu_bundle(arrays, graph, device, chunk=256):
    validate_graph(graph, len(arrays["h96"]), arrays.get("roi"))
    data = {
        key: torch.as_tensor(np.array(arrays[key], copy=True), dtype=torch.float32, device=device)
        for key in ("h96", "h192", "tier", "tissue")
    }
    with torch.no_grad():
        data["node_features"] = torch.cat(
            (
                f.layer_norm(data["h96"], (1536,)),
                f.layer_norm(data["h192"], (1536,)),
                data["tier"],
                data["tissue"],
            ),
            dim=1,
        )
        neighbors = torch.as_tensor(graph["neighbors"], dtype=torch.long, device=device)
        weights = torch.as_tensor(graph["weights"], dtype=torch.float32, device=device)
        means = torch.empty_like(data["node_features"])
        for left in range(0, len(neighbors), chunk):
            right = min(left + chunk, len(neighbors))
            means[left:right] = (
                data["node_features"][neighbors[left:right].clamp_min(0)]
                * weights[left:right].unsqueeze(-1)
            ).sum(1)
        data.update(neighbors=neighbors, weights=weights, neighbor_mean=means)
    return data


def validate_graph(graph, n, rois=None):
    neighbors, weights = np.asarray(graph["neighbors"]), np.asarray(graph["weights"])
    if neighbors.ndim != 2 or neighbors.shape[0] != n or weights.shape != neighbors.shape:
        raise ValueError("Graph neighbors and weights must have matching N x K shapes")
    if neighbors.dtype.kind not in "iu" or np.any((neighbors < -1) | (neighbors >= n)):
        raise ValueError("Graph contains invalid neighbor indices")
    valid = neighbors >= 0
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights[~valid] != 0):
        raise ValueError("Graph weights must be finite, nonnegative and zero on missing neighbors")
    totals = weights.sum(1)
    if not np.allclose(totals[valid.any(1)], 1, atol=1e-5):
        raise ValueError("Graph weights must sum to one for each connected nucleus")
    rows = np.broadcast_to(np.arange(n)[:, None], neighbors.shape)
    if np.any(neighbors[valid] == rows[valid]):
        raise ValueError("Graph must exclude self neighbors")
    if rois is not None and np.any(
        np.asarray(rois)[rows[valid]] != np.asarray(rois)[neighbors[valid]]
    ):
        raise ValueError("Graph contains cross-ROI edges, which can leak validation data")
    if "distance" in graph:
        distance = np.asarray(graph["distance"])
        if (
            distance.shape != neighbors.shape
            or not np.isfinite(distance).all()
            or np.any(distance < 0)
        ):
            raise ValueError("Graph distances must be finite, nonnegative and match neighbors")
