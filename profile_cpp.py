import argparse
from time import perf_counter

import numpy as np

from web_app import make_overlap_blobs_nd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--points-per-cluster", type=int, default=650)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=35)
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--init", type=str, default="kmeans++", choices=("kmeans++", "random"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        import kmeans_cpp
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "kmeans_cpp extension not importable. Build it first with:\n"
            "  python -m pip install -r requirements.txt\n"
            "  python -m pip install -e ."
        ) from e

    X = make_overlap_blobs_nd(
        seed=args.seed,
        n_clusters=args.k,
        points_per_cluster=args.points_per_cluster,
        n_features=args.n_features,
        center_scale=10.0,
        cluster_std=1.2,
    )

    t0 = perf_counter()
    res = kmeans_cpp.kmeans(
        X,
        int(args.k),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        init=str(args.init),
        n_init=int(args.n_init),
        random_state=int(args.seed),
        profile=True,
    )
    elapsed = perf_counter() - t0

    timing = res.get("timing")
    print(f"elapsed_s: {elapsed:.6f}")
    if isinstance(timing, dict):
        for key in ("validate_s", "init_s", "assign_s", "update_s", "total_s"):
            if key in timing:
                print(f"{key}: {float(timing[key]):.6f}")


if __name__ == "__main__":
    main()
