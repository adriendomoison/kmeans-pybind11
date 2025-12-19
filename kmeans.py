from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional

import numpy as np


@dataclass
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int
    timing: Optional[dict[str, float]] = None


def _validate_inputs(X: np.ndarray, n_clusters: int, max_iter: int, tol: float) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
    if X.shape[0] == 0:
        raise ValueError("X must have at least one sample")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be >= 1")
    if n_clusters > X.shape[0]:
        raise ValueError("n_clusters must be <= n_samples")
    if max_iter <= 0:
        raise ValueError("max_iter must be >= 1")
    if tol < 0:
        raise ValueError("tol must be >= 0")
    return X


def _init_centers(X: np.ndarray, n_clusters: int, rng: np.random.Generator, init: str) -> np.ndarray:
    if init == "random":
        idx = rng.choice(X.shape[0], size=n_clusters, replace=False)
        return X[idx].copy()

    if init != "kmeans++":
        raise ValueError("init must be 'kmeans++' or 'random'")

    n, _ = X.shape
    centers = np.empty((n_clusters, X.shape[1]), dtype=X.dtype)

    first = int(rng.integers(0, n))
    centers[0] = X[first]

    closest_dist2 = np.sum((X - centers[0]) ** 2, axis=1)

    for c in range(1, n_clusters):
        probs = closest_dist2 / np.sum(closest_dist2)
        next_idx = int(rng.choice(n, p=probs))
        centers[c] = X[next_idx]
        dist2_new = np.sum((X - centers[c]) ** 2, axis=1)
        closest_dist2 = np.minimum(closest_dist2, dist2_new)

    return centers


def _assign_numpy(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, float]:
    diff = X[:, None, :] - centers[None, :, :]
    dist2 = np.einsum("nkd,nkd->nk", diff, diff)
    labels = np.argmin(dist2, axis=1).astype(np.int32)
    inertia = float(np.sum(dist2[np.arange(X.shape[0]), labels]))
    return labels, inertia


def _assign_naive(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, float]:
    n = X.shape[0]
    k = centers.shape[0]
    labels = np.empty(n, dtype=np.int32)
    inertia = 0.0
    for i in range(n):
        best_k = 0
        best_d2 = float("inf")
        xi = X[i]
        for c in range(k):
            diff = xi - centers[c]
            d2 = float(np.dot(diff, diff))
            if d2 < best_d2:
                best_d2 = d2
                best_k = c
        labels[i] = best_k
        inertia += best_d2
    return labels, float(inertia)


def _update_numpy(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    new_centers = np.empty((n_clusters, X.shape[1]), dtype=X.dtype)
    for k in range(n_clusters):
        mask = labels == k
        if not np.any(mask):
            new_centers[k] = X[int(rng.integers(0, X.shape[0]))]
        else:
            new_centers[k] = X[mask].mean(axis=0)
    return new_centers


def _update_naive(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    sums = np.zeros((n_clusters, X.shape[1]), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)

    for i in range(X.shape[0]):
        c = int(labels[i])
        sums[c] += X[i]
        counts[c] += 1

    new_centers = np.empty((n_clusters, X.shape[1]), dtype=np.float64)
    for c in range(n_clusters):
        if counts[c] == 0:
            new_centers[c] = X[int(rng.integers(0, X.shape[0]))]
        else:
            new_centers[c] = sums[c] / float(counts[c])
    return new_centers


def kmeans(
    X: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int = 300,
    tol: float = 1e-4,
    init: str = "kmeans++",
    algorithm: str = "numpy",
    n_init: int = 1,
    random_state: Optional[int] = None,
    callback: Optional[Callable[[int, float, float], None]] = None,
    profile: bool = False,
) -> KMeansResult:
    t_total_start = perf_counter() if profile else 0.0

    t_validate_start = perf_counter() if profile else 0.0
    X = _validate_inputs(X, n_clusters=n_clusters, max_iter=max_iter, tol=tol)
    t_validate = (perf_counter() - t_validate_start) if profile else 0.0

    rng = np.random.default_rng(random_state)

    if algorithm not in ("numpy", "naive"):
        raise ValueError("algorithm must be 'numpy' or 'naive'")

    best_inertia = float("inf")
    best_labels: Optional[np.ndarray] = None
    best_centers: Optional[np.ndarray] = None
    best_n_iter = 0

    t_assign_total = 0.0
    t_update_total = 0.0

    for _ in range(int(n_init)):
        centers = _init_centers(X, n_clusters=n_clusters, rng=rng, init=init)

        labels = np.zeros(X.shape[0], dtype=np.int32)
        inertia = float("inf")

        for it in range(int(max_iter)):
            t_assign_start = perf_counter() if profile else 0.0
            if algorithm == "numpy":
                labels, inertia = _assign_numpy(X, centers)
            else:
                labels, inertia = _assign_naive(X, centers)
            if profile:
                t_assign_total += perf_counter() - t_assign_start

            t_update_start = perf_counter() if profile else 0.0
            if algorithm == "numpy":
                new_centers = _update_numpy(X, labels, n_clusters, rng=rng)
            else:
                new_centers = _update_naive(X, labels, n_clusters, rng=rng)
            if profile:
                t_update_total += perf_counter() - t_update_start

            shift = float(np.sqrt(np.sum((new_centers - centers) ** 2)))
            centers = new_centers

            if callback is not None:
                callback(int(it), float(inertia), float(shift))

            if shift <= tol:
                it += 1
                break

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_n_iter = it

    timing: Optional[dict[str, float]] = None
    if profile:
        timing = {
            "validate_s": t_validate,
            "assign_s": t_assign_total,
            "update_s": t_update_total,
            "total_s": perf_counter() - t_total_start,
        }

    return KMeansResult(
        labels=np.asarray(best_labels),
        centers=np.asarray(best_centers),
        inertia=float(best_inertia),
        n_iter=int(best_n_iter),
        timing=timing,
    )


class KMeans:
    def __init__(
        self,
        n_clusters: int = 8,
        *,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: str = "kmeans++",
        n_init: int = 1,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.init = str(init)
        self.n_init = int(n_init)
        self.random_state = random_state

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "KMeans":
        result = kmeans(
            X,
            self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            init=self.init,
            n_init=self.n_init,
            random_state=self.random_state,
            profile=False,
        )
        self.cluster_centers_ = result.centers
        self.labels_ = result.labels
        self.inertia_ = result.inertia
        self.n_iter_ = result.n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError("KMeans.predict called before fit")
        X = np.asarray(X)
        diff = X[:, None, :] - self.cluster_centers_[None, :, :]
        dist2 = np.einsum("nkd,nkd->nk", diff, diff)
        return np.argmin(dist2, axis=1).astype(np.int32)
