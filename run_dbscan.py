import numpy as np
import matplotlib.pyplot as plt
import argparse
from time import perf_counter
import sys

from dbscan import dbscan, NOISE


def make_data(seed: int = 0):
    # Simple 2D dataset: a few blobs + uniform noise.
    rng = np.random.default_rng(seed)

    c1 = rng.normal(loc=(0.0, 0.0), scale=0.22, size=(250, 2))
    c2 = rng.normal(loc=(2.0, 1.2), scale=0.18, size=(220, 2))
    c3 = rng.normal(loc=(-1.6, 1.4), scale=0.20, size=(210, 2))

    noise = rng.uniform(low=(-3.0, -2.0), high=(3.0, 2.5), size=(120, 2))

    X = np.vstack([c1, c2, c3, noise])
    return X


def make_fireworks_data(
    seed: int = 0,
    *,
    n_bursts: int = 5,
    spokes_per_burst: int = 11,
    points_per_spoke: int = 55,
):
    rng = np.random.default_rng(seed)

    centers = rng.normal(loc=(0.0, 0.0), scale=1.15, size=(n_bursts, 2))
    chunks = []

    for c in centers:
        base_angles = np.linspace(0.0, 2.0 * np.pi, spokes_per_burst, endpoint=False)
        angles = base_angles + rng.normal(scale=0.08, size=spokes_per_burst)
        for theta in angles:
            direction = np.array([np.cos(theta), np.sin(theta)])
            perp = np.array([-np.sin(theta), np.cos(theta)])

            r = rng.uniform(0.25, 3.2, size=points_per_spoke)
            r += rng.normal(scale=0.04, size=points_per_spoke)

            radial_jitter = rng.normal(scale=0.03, size=points_per_spoke)
            thickness = 0.06 + 0.04 * r
            perp_jitter = rng.normal(scale=1.0, size=points_per_spoke) * thickness

            pts = c + (r + radial_jitter)[:, None] * direction + perp_jitter[:, None] * perp
            chunks.append(pts)

    X = np.vstack(chunks)
    noise = rng.uniform(low=(-4.0, -3.5), high=(4.0, 3.8), size=(260, 2))
    X = np.vstack([X, noise])
    return X


def plot_before(X: np.ndarray, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], s=18, c="slategray", alpha=0.65)
    ax.axis("equal")
    ax.grid(True, linewidth=0.3, alpha=0.5)


def plot_clusters(X: np.ndarray, labels: np.ndarray, core_mask: np.ndarray, ax=None):
    if ax is None:
        ax = plt.gca()
    unique = np.unique(labels)

    try:
        cmap = plt.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")

    for idx, k in enumerate(unique):
        mask = labels == k
        if k == NOISE:
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                s=15,
                c="lightgray",
                marker="x",
                linewidths=0.8,
                label="noise",
            )
            continue

        color = cmap(idx % cmap.N)
        ax.scatter(X[mask, 0], X[mask, 1], s=18, color=color, alpha=0.75, label=f"cluster {k}")

        core = mask & core_mask
        ax.scatter(
            X[core, 0],
            X[core, 1],
            s=50,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
        )

    ax.axis("equal")
    ax.grid(True, linewidth=0.3, alpha=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=("blobs", "fireworks"), default="fireworks")
    parser.add_argument("--impl", type=str, choices=("python", "cpp", "both"), default="python")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    if args.dataset == "blobs":
        X = make_data(seed=7)
        eps = 0.35
        min_samples = 8
    else:
        X = make_fireworks_data(seed=7, n_bursts=5)
        eps = 0.22
        min_samples = 7

    result_py = None
    result_cpp = None

    if args.impl in ("python", "both"):
        t0 = perf_counter()
        result_py = dbscan(X, eps=eps, min_samples=min_samples, profile=True)
        elapsed = perf_counter() - t0

        print(f"python dbscan(...) elapsed: {elapsed:.6f} s")
        if result_py.timing is not None:
            print("python timing breakdown (s):")
            for k in ("validate_s", "neighbor_queries_s", "expand_loop_s", "total_s"):
                if k in result_py.timing:
                    print(f"  {k}: {result_py.timing[k]:.6f}")

    if args.impl in ("cpp", "both"):
        try:
            import dbscan_cpp
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "C++ extension not importable. This usually means it was built for a different Python version.\n"
                f"Current interpreter: {sys.executable} (Python {sys.version.split()[0]})\n"
                "Build/install with the *same* interpreter, e.g.:\n"
                "  python -m pip install -r requirements.txt\n"
                "  python -m pip install -e ."
            ) from e

        t0 = perf_counter()
        result_cpp = dbscan_cpp.dbscan(X, eps=eps, min_samples=min_samples, profile=True)
        elapsed = perf_counter() - t0

        print(f"cpp dbscan(...) elapsed: {elapsed:.6f} s")
        if result_cpp.get("timing") is not None:
            print("cpp timing breakdown (s):")
            for k in ("validate_s", "neighbor_queries_s", "expand_loop_s", "total_s"):
                if k in result_cpp["timing"]:
                    print(f"  {k}: {result_cpp['timing'][k]:.6f}")

    if args.impl == "cpp":
        labels = result_cpp["labels"]
        core_mask = result_cpp["core_sample_mask"]
    else:
        labels = result_py.labels
        core_mask = result_py.core_sample_mask

    if args.impl == "both":
        same_labels = np.array_equal(result_py.labels, result_cpp["labels"])
        same_core = np.array_equal(result_py.core_sample_mask, result_cpp["core_sample_mask"])
        if not (same_labels and same_core):
            print("WARNING: python and cpp outputs differ")

    if args.compare:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        plot_before(X, ax=ax0)
        ax0.set_title("Before: raw data")

        plot_clusters(X, labels, core_mask, ax=ax1)
        ax1.set_title(f"After: DBSCAN (eps={eps}, min_samples={min_samples})")
        ax1.legend(loc="best", fontsize=9)
    else:
        fig = plt.figure(figsize=(9, 6))
        ax = plt.gca()
        plot_clusters(X, labels, core_mask, ax=ax)
        ax.set_title(f"DBSCAN from scratch (eps={eps}, min_samples={min_samples})")
        ax.legend(loc="best", fontsize=9)

    if args.save:
        fig.savefig(args.save, dpi=160, bbox_inches="tight")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
