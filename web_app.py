from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kmeans import kmeans


app = FastAPI(title="K-means Python vs C++ (pybind11)")

_BASE_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _BASE_DIR / "static"


class RunRequest(BaseModel):
    impl: Literal["python", "cpp"] = "python"
    preset: Literal["bench"] = "bench"

    k: Optional[int] = Field(default=None, ge=2)


def make_overlap_blobs(seed: int = 0, *, n_clusters: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, num=int(n_clusters), endpoint=False)
    radius = 7.0
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    centers = centers + rng.normal(scale=0.35, size=centers.shape)

    chunks = []
    cluster_std = 0.65
    cov = (cluster_std**2) * np.eye(2)
    for c in centers:
        pts = rng.multivariate_normal(mean=c, cov=cov, size=260)
        chunks.append(pts)

    X = np.vstack(chunks)
    X = np.vstack([X, rng.uniform(low=(-11.0, -10.0), high=(11.0, 10.0), size=(140, 2))])
    return X


def make_overlap_blobs_nd(
    seed: int = 0,
    *,
    n_clusters: int,
    points_per_cluster: int,
    n_features: int,
    center_scale: float,
    cluster_std: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_clusters = int(n_clusters)
    n_features = int(n_features)
    if n_features < 2:
        raise ValueError("n_features must be >= 2")

    angles = np.linspace(0.0, 2.0 * np.pi, num=n_clusters, endpoint=False)
    radius = float(center_scale) * np.sqrt(2.0)
    centers = np.zeros((n_clusters, n_features), dtype=np.float64)
    centers[:, 0] = radius * np.cos(angles)
    centers[:, 1] = radius * np.sin(angles)
    centers[:, :2] += rng.normal(scale=0.25 * float(center_scale), size=(n_clusters, 2))

    if n_features > 2:
        extra_scale = 0.35 * float(center_scale)
        centers[:, 2:] = rng.normal(scale=extra_scale, size=(n_clusters, n_features - 2))

    chunks = []
    cov = (float(cluster_std) ** 2) * np.eye(int(n_features))
    for c in centers:
        pts = rng.multivariate_normal(mean=c, cov=cov, size=int(points_per_cluster))
        chunks.append(pts)

    X = np.vstack(chunks)
    return X


def _bench_defaults() -> dict:
    return {
        "k": 25,
        "points_per_cluster": 650,
        "n_features": 10,
        "max_iter": 35,
        "n_init": 1,
        "tol": 1e-4,
        "init": "kmeans++",
        "random_state": 7,
        "python_algo": "naive",
    }


def _quick_defaults() -> dict:
    return {
        "k": 5,
        "max_iter": 120,
        "n_init": 3,
        "tol": 1e-4,
        "init": "kmeans++",
        "random_state": 7,
        "python_algo": "numpy",
    }


def _make_dataset(req: RunRequest) -> tuple[np.ndarray, int, dict]:
    cfg = _bench_defaults()
    k = int(req.k) if req.k is not None else int(cfg["k"])
    seed = int(cfg["random_state"])
    points_per_cluster = int(cfg["points_per_cluster"])
    n_features = int(cfg["n_features"])

    X = make_overlap_blobs_nd(
        seed=seed,
        n_clusters=k,
        points_per_cluster=points_per_cluster,
        n_features=n_features,
        center_scale=10.0,
        cluster_std=1.2,
    )
    return X, k, cfg


def _downsample_for_plot(X: np.ndarray, labels: np.ndarray, *, plot_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(X.shape[0])
    if plot_points >= n:
        return X[:, :2], labels

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(plot_points), replace=False)
    idx.sort()
    return X[idx, :2], labels[idx]


@app.get("/")
def index():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/run")
def run(req: RunRequest):
    X, k, cfg = _make_dataset(req)

    max_iter = int(cfg["max_iter"])
    n_init = int(cfg["n_init"])
    tol = float(cfg["tol"])
    init = str(cfg["init"])
    random_state = int(cfg["random_state"])
    py_algo = str(cfg["python_algo"])

    def _run_impl() -> dict:
        if req.impl == "python":
            return _run_python()
        if req.impl == "cpp":
            return _run_cpp()
        raise HTTPException(status_code=400, detail="impl must be 'python' or 'cpp'")

    def _run_python() -> dict:
        t0 = perf_counter()
        res = kmeans(
            X,
            int(k),
            max_iter=max_iter,
            tol=tol,
            init=init,
            algorithm=py_algo,
            n_init=n_init,
            random_state=random_state,
            profile=True,
        )
        elapsed = perf_counter() - t0
        X2, y = _downsample_for_plot(X, res.labels, plot_points=5000, seed=random_state)
        return {
            "elapsed_s": float(elapsed),
            "plot": {
                "X2": X2.tolist(),
                "labels": y.astype(np.int32).tolist(),
                "centers2": res.centers[:, :2].tolist(),
            },
        }

    def _run_cpp() -> dict:
        try:
            import kmeans_cpp
        except ModuleNotFoundError as e:
            raise HTTPException(
                status_code=500,
                detail=(
                    "C++ extension not importable on the server. "
                    "The deployment must build the pybind11 extension (pip install -e .)."
                ),
            ) from e

        t0 = perf_counter()
        res = kmeans_cpp.kmeans(
            X,
            int(k),
            max_iter=max_iter,
            tol=tol,
            init=init,
            n_init=n_init,
            random_state=random_state,
            profile=True,
        )
        elapsed = perf_counter() - t0
        labels = np.asarray(res["labels"], dtype=np.int32)
        centers = np.asarray(res["centers"], dtype=np.float64)
        X2, y = _downsample_for_plot(X, labels, plot_points=5000, seed=random_state)
        return {
            "elapsed_s": float(elapsed),
            "plot": {
                "X2": X2.tolist(),
                "labels": y.astype(np.int32).tolist(),
                "centers2": centers[:, :2].tolist(),
            },
        }

    return _run_impl()


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
