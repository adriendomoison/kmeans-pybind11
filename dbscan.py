from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Optional

import numpy as np


# Sentinel labels used by DBSCAN.
UNCLASSIFIED = -2
NOISE = -1


@dataclass
class DBSCANResult:
    labels: np.ndarray
    core_sample_mask: np.ndarray
    timing: Optional[dict[str, float]] = None


def _validate_inputs(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
    if X.shape[0] == 0:
        raise ValueError("X must have at least one sample")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if min_samples <= 0:
        raise ValueError("min_samples must be >= 1")
    return X


def _neighbors_within_eps(X: np.ndarray, index: int, eps: float) -> np.ndarray:
    # Brute-force ε-neighborhood query.
    diff = X - X[index]
    dist2 = np.einsum("ij,ij->i", diff, diff)
    return np.flatnonzero(dist2 <= eps * eps)


def dbscan(X: np.ndarray, eps: float, min_samples: int, *, profile: bool = False) -> DBSCANResult:
    t_total_start = perf_counter() if profile else 0.0

    t_validate_start = perf_counter() if profile else 0.0
    X = _validate_inputs(X, eps=eps, min_samples=min_samples)
    t_validate = (perf_counter() - t_validate_start) if profile else 0.0

    n = X.shape[0]

    labels = np.full(n, UNCLASSIFIED, dtype=int)

    visited = np.zeros(n, dtype=bool)

    core_sample_mask = np.zeros(n, dtype=bool)

    cluster_id = 0

    t_neighbor_queries = 0.0
    t_expand_loop = 0.0

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True

        t_neighbors_start = perf_counter() if profile else 0.0
        neighbors = _neighbors_within_eps(X, i, eps)
        if profile:
            t_neighbor_queries += perf_counter() - t_neighbors_start

        if neighbors.size < min_samples:
            labels[i] = NOISE
            continue

        core_sample_mask[i] = True
        labels[i] = cluster_id

        seeds = deque(int(j) for j in neighbors if int(j) != i)
        in_seeds = set(int(j) for j in neighbors)
        in_seeds.discard(int(i))

        t_expand_start = perf_counter() if profile else 0.0
        while seeds:
            j = seeds.popleft()
            in_seeds.discard(j)

            if not visited[j]:
                visited[j] = True

                t_neighbors_j_start = perf_counter() if profile else 0.0
                neighbors_j = _neighbors_within_eps(X, j, eps)
                if profile:
                    t_neighbor_queries += perf_counter() - t_neighbors_j_start

                if neighbors_j.size >= min_samples:
                    core_sample_mask[j] = True
                    for k in neighbors_j.tolist():
                        k = int(k)

                        if labels[k] == UNCLASSIFIED:
                            labels[k] = cluster_id

                        if not visited[k] and k not in in_seeds:
                            seeds.append(k)
                            in_seeds.add(k)

            if labels[j] in (UNCLASSIFIED, NOISE):
                labels[j] = cluster_id

        if profile:
            t_expand_loop += perf_counter() - t_expand_start

        cluster_id += 1

    labels[labels == UNCLASSIFIED] = NOISE

    timing: Optional[dict[str, float]] = None
    if profile:
        timing = {
            "validate_s": t_validate,
            "neighbor_queries_s": t_neighbor_queries,
            "expand_loop_s": t_expand_loop,
            "total_s": perf_counter() - t_total_start,
        }

    return DBSCANResult(labels=labels, core_sample_mask=core_sample_mask, timing=timing)


class DBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.labels_: Optional[np.ndarray] = None
        self.core_sample_mask_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DBSCAN":
        result = dbscan(X, eps=self.eps, min_samples=self.min_samples)
        self.labels_ = result.labels
        self.core_sample_mask_ = result.core_sample_mask
        return self
