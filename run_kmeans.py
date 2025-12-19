import argparse
import itertools
import threading
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from kmeans import kmeans


def make_overlap_blobs(seed: int = 0, *, n_clusters: int = 5):
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
):
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


def plot_before(X: np.ndarray, ax=None):
    if ax is None:
        ax = plt.gca()
    X2 = X[:, :2] if X.shape[1] > 2 else X
    ax.scatter(X2[:, 0], X2[:, 1], s=14, c="slategray", alpha=0.55)
    ax.axis("equal")
    ax.grid(True, linewidth=0.3, alpha=0.5)


def plot_after(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, ax=None):
    if ax is None:
        ax = plt.gca()

    X2 = X[:, :2] if X.shape[1] > 2 else X
    C2 = centers[:, :2] if centers.shape[1] > 2 else centers

    unique = np.unique(labels)
    try:
        cmap = plt.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")

    for idx, k in enumerate(unique):
        mask = labels == k
        color = cmap(idx % cmap.N)
        ax.scatter(X2[mask, 0], X2[mask, 1], s=14, color=color, alpha=0.7)

    ax.scatter(C2[:, 0], C2[:, 1], s=160, c="black", marker="x", linewidths=1.3)
    ax.axis("equal")
    ax.grid(True, linewidth=0.3, alpha=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", type=str, choices=("python", "cpp", "both"), default="python")
    parser.add_argument("--preset", type=str, choices=("quick", "bench"), default="quick")
    parser.add_argument("--python-algo", type=str, choices=("numpy", "naive"), default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--points-per-cluster", type=int, default=None)
    parser.add_argument("--n-features", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    if args.preset == "quick":
        n_clusters = 5
        X = make_overlap_blobs(seed=7, n_clusters=n_clusters)
        max_iter = 120
        tol = 1e-4
        init = "kmeans++"
        n_init = 3
        random_state = 7
        python_algo = args.python_algo or "numpy"
    else:
        n_clusters = int(args.k) if args.k is not None else 25
        points_per_cluster = int(args.points_per_cluster) if args.points_per_cluster is not None else 650
        n_features = int(args.n_features) if args.n_features is not None else 10
        X = make_overlap_blobs_nd(
            seed=7,
            n_clusters=n_clusters,
            points_per_cluster=points_per_cluster,
            n_features=n_features,
            center_scale=10.0,
            cluster_std=1.2,
        )
        max_iter = int(args.max_iter) if args.max_iter is not None else 35
        tol = 1e-4
        init = "kmeans++"
        n_init = int(args.n_init) if args.n_init is not None else 1
        random_state = 7
        python_algo = args.python_algo or "naive"

    live = bool(args.live and not args.no_show)
    if live:
        plt.ion()

    result_py = None
    result_cpp = None

    fig_py = None
    ax_py_before = None
    ax_py_after = None
    py_start = 0.0

    def _python_live_callback(it: int, inertia: float, shift: float):
        if not live or fig_py is None:
            return
        elapsed = perf_counter() - py_start
        fig_py.suptitle(f"Python K-means running... {elapsed:.2f}s (iter {it+1}/{max_iter})")
        fig_py.canvas.draw_idle()
        plt.pause(0.001)

    if args.impl in ("python", "both"):
        if live:
            fig_py, (ax_py_before, ax_py_after) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            plot_before(X, ax=ax_py_before)
            ax_py_before.set_title("Before")
            ax_py_after.set_title("After (pending)")
            fig_py.suptitle("Python K-means running... 0.00s")
            fig_py.canvas.draw_idle()
            plt.pause(0.001)
            py_start = perf_counter()

        t0 = perf_counter()
        result_py = kmeans(
            X,
            n_clusters,
            max_iter=max_iter,
            tol=tol,
            init=init,
            algorithm=python_algo,
            n_init=n_init,
            random_state=random_state,
            profile=True,
            callback=_python_live_callback if live else None,
        )
        elapsed = perf_counter() - t0
        print(f"python kmeans(...) elapsed: {elapsed:.6f} s")
        if result_py.timing is not None:
            print("python timing breakdown (s):")
            for k in ("validate_s", "assign_s", "update_s", "total_s"):
                if k in result_py.timing:
                    print(f"  {k}: {result_py.timing[k]:.6f}")
        print(f"python inertia: {result_py.inertia:.6f} (n_iter={result_py.n_iter})")

        if live:
            plot_after(X, result_py.labels, result_py.centers, ax=ax_py_after)
            ax_py_after.set_title("After")
            fig_py.suptitle(f"Python K-means done in {elapsed:.2f}s")
            fig_py.canvas.draw_idle()
            plt.pause(0.001)

    fig_cpp = None
    ax_cpp_before = None
    ax_cpp_after = None
    result_cpp_holder = {"value": None, "elapsed": None, "error": None}

    def _run_cpp():
        try:
            import kmeans_cpp

            t0 = perf_counter()
            out = kmeans_cpp.kmeans(
                X,
                n_clusters,
                max_iter=max_iter,
                tol=tol,
                init=init,
                n_init=n_init,
                random_state=random_state,
                profile=True,
            )
            result_cpp_holder["value"] = out
            result_cpp_holder["elapsed"] = perf_counter() - t0
        except BaseException as e:
            result_cpp_holder["error"] = e

    if args.impl in ("cpp", "both"):
        try:
            import kmeans_cpp
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "C++ extension not importable. Build it with:\n"
                "  python -m pip install -r requirements.txt\n"
                "  python -m pip install -e ."
            ) from e

        if live:
            fig_cpp, (ax_cpp_before, ax_cpp_after) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            plot_before(X, ax=ax_cpp_before)
            ax_cpp_before.set_title("Before")
            ax_cpp_after.set_title("After (pending)")
            fig_cpp.suptitle("C++ K-means running... 0.00s")
            fig_cpp.canvas.draw_idle()
            plt.pause(0.001)

            t_start = perf_counter()
            th = threading.Thread(target=_run_cpp, daemon=True)
            th.start()
            while th.is_alive():
                elapsed = perf_counter() - t_start
                fig_cpp.suptitle(f"C++ K-means running... {elapsed:.2f}s")
                fig_cpp.canvas.draw_idle()
                plt.pause(0.05)
            th.join()

            if result_cpp_holder["error"] is not None:
                raise result_cpp_holder["error"]
            result_cpp = result_cpp_holder["value"]
            elapsed = float(result_cpp_holder["elapsed"])
        else:
            t0 = perf_counter()
            result_cpp = kmeans_cpp.kmeans(
                X,
                n_clusters,
                max_iter=max_iter,
                tol=tol,
                init=init,
                n_init=n_init,
                random_state=random_state,
                profile=True,
            )
            elapsed = perf_counter() - t0

        print(f"cpp kmeans(...) elapsed: {elapsed:.6f} s")
        if result_cpp.get("timing") is not None:
            print("cpp timing breakdown (s):")
            for k in ("validate_s", "init_s", "assign_s", "update_s", "total_s"):
                if k in result_cpp["timing"]:
                    print(f"  {k}: {result_cpp['timing'][k]:.6f}")
        print(f"cpp inertia: {float(result_cpp['inertia']):.6f} (n_iter={int(result_cpp['n_iter'])})")

        if live:
            plot_after(X, result_cpp["labels"], result_cpp["centers"], ax=ax_cpp_after)
            ax_cpp_after.set_title("After")
            fig_cpp.suptitle(f"C++ K-means done in {elapsed:.2f}s")
            fig_cpp.canvas.draw_idle()
            plt.pause(0.001)

    labels_py = result_py.labels if result_py is not None else None
    centers_py = result_py.centers if result_py is not None else None
    inertia_py = float(result_py.inertia) if result_py is not None else None
    n_iter_py = int(result_py.n_iter) if result_py is not None else None

    labels_cpp = result_cpp["labels"] if result_cpp is not None else None
    centers_cpp = result_cpp["centers"] if result_cpp is not None else None
    inertia_cpp = float(result_cpp["inertia"]) if result_cpp is not None else None
    n_iter_cpp = int(result_cpp["n_iter"]) if result_cpp is not None else None

    if args.impl == "both" and args.check:
        dist2 = np.sum((centers_py[:, None, :] - centers_cpp[None, :, :]) ** 2, axis=2)
        best_perm = None
        best_cost = float("inf")
        for perm in itertools.permutations(range(n_clusters)):
            cost = float(np.sum(dist2[np.arange(n_clusters), np.array(perm)]))
            if cost < best_cost:
                best_cost = cost
                best_perm = perm

        perm_arr = np.array(best_perm, dtype=np.int32)
        mapped_labels_py = perm_arr[labels_py]
        max_center_err = float(np.sqrt(np.max(dist2[np.arange(n_clusters), perm_arr])))

        same_labels = np.array_equal(mapped_labels_py, labels_cpp)
        same_centers = np.allclose(centers_py, centers_cpp[np.array(best_perm)], atol=1e-6, rtol=1e-6)

        if not (same_labels and same_centers):
            print(
                "WARNING: python and cpp outputs differ "
                f"(best center matching max err={max_center_err:.3e}, inertia_py={inertia_py:.6f}, inertia_cpp={inertia_cpp:.6f})"
            )

    if args.impl == "python":
        plot_specs = [("Python", labels_py, centers_py, inertia_py, n_iter_py)]
    elif args.impl == "cpp":
        plot_specs = [("C++", labels_cpp, centers_cpp, inertia_cpp, n_iter_cpp)]
    else:
        plot_specs = [
            ("Python", labels_py, centers_py, inertia_py, n_iter_py),
            ("C++", labels_cpp, centers_cpp, inertia_cpp, n_iter_cpp),
        ]

    if live:
        if args.save:
            if fig_py is not None:
                fig_py.savefig(f"py_{args.save}", dpi=160, bbox_inches="tight")
            if fig_cpp is not None:
                fig_cpp.savefig(f"cpp_{args.save}", dpi=160, bbox_inches="tight")
        plt.ioff()
        plt.show()
        return

    if args.compare and args.impl == "both":
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        plot_before(X, ax=ax0)
        ax0.set_title("Before: raw data")

        (name1, lab1, cen1, iner1, it1) = plot_specs[0]
        plot_after(X, lab1, cen1, ax=ax1)
        ax1.set_title(f"After ({name1}): k={n_clusters}, inertia={iner1:.3f}, iters={it1}")

        (name2, lab2, cen2, iner2, it2) = plot_specs[1]
        plot_after(X, lab2, cen2, ax=ax2)
        ax2.set_title(f"After ({name2}): k={n_clusters}, inertia={iner2:.3f}, iters={it2}")
    elif args.compare:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        plot_before(X, ax=ax0)
        ax0.set_title("Before: raw data")

        (name, lab, cen, iner, it) = plot_specs[0]
        plot_after(X, lab, cen, ax=ax1)
        ax1.set_title(f"After ({name}): k={n_clusters}, inertia={iner:.3f}, iters={it}")
    elif args.impl == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        (name1, lab1, cen1, iner1, it1) = plot_specs[0]
        plot_after(X, lab1, cen1, ax=ax1)
        ax1.set_title(f"After ({name1}): k={n_clusters}, inertia={iner1:.3f}, iters={it1}")

        (name2, lab2, cen2, iner2, it2) = plot_specs[1]
        plot_after(X, lab2, cen2, ax=ax2)
        ax2.set_title(f"After ({name2}): k={n_clusters}, inertia={iner2:.3f}, iters={it2}")
    else:
        fig = plt.figure(figsize=(9, 6))
        ax = plt.gca()
        (name, lab, cen, iner, it) = plot_specs[0]
        plot_after(X, lab, cen, ax=ax)
        ax.set_title(f"After ({name}): k={n_clusters}, inertia={iner:.3f}, iters={it}")

    if args.save:
        fig.savefig(args.save, dpi=160, bbox_inches="tight")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
