import argparse
import cProfile
import pstats
from pathlib import Path
from time import perf_counter

import numpy as np

from kmeans import kmeans
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
    parser.add_argument("--python-algo", type=str, default="naive", choices=("numpy", "naive"))
    parser.add_argument("--sort", type=str, default="tottime")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    X = make_overlap_blobs_nd(
        seed=args.seed,
        n_clusters=args.k,
        points_per_cluster=args.points_per_cluster,
        n_features=args.n_features,
        center_scale=10.0,
        cluster_std=1.2,
    )

    def run_once() -> None:
        _ = kmeans(
            X,
            args.k,
            max_iter=args.max_iter,
            tol=args.tol,
            init=args.init,
            algorithm=args.python_algo,
            n_init=args.n_init,
            random_state=args.seed,
            profile=False,
        )

    pr = cProfile.Profile()
    t0 = perf_counter()
    pr.enable()
    run_once()
    pr.disable()
    elapsed = perf_counter() - t0

    print(f"elapsed_s: {elapsed:.6f}")

    stats = pstats.Stats(pr).strip_dirs().sort_stats(args.sort)
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stats.dump_stats(str(out_path))
        print(f"wrote_profile: {out_path}")

    stats.print_stats(args.limit)


if __name__ == "__main__":
    main()
