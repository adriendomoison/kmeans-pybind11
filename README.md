---
title: K-means Python vs C++ (pybind11)
sdk: docker
---

This repository revisits and modernizes a k-means clustering project originally implemented during my EPITECH mathematics coursework, adapted here to demonstrate a hybrid Python/C++ (pybind11) approach for performance-critical computation.

# K-means: Python vs C++ (pybind11)

## Purpose

This repository presents a hybrid Python/C++ architecture for performance-critical scientific workloads:

- Python is used for orchestration, configuration, and experiment iteration.
- C++ is used for the computational kernel (k-means inner loop).
- `pybind11` provides a thin binding layer so the C++ kernel can be called from Python.

The intent is to make the boundary between experimentation code and performance-sensitive kernels explicit.

## Architecture overview

```
┌──────────────────────────────┐
│ Python layer                  │
│  - dataset generation         │
│  - experiment orchestration   │
│  - web API + UI integration   │
│  Files: web_app.py, kmeans.py │
└───────────────┬──────────────┘
                │ (NumPy arrays)
┌───────────────▼──────────────┐
│ Binding layer (pybind11)      │
│  - marshaling + validation    │
│  Files: setup.py, kmeans_cpp  │
└───────────────┬──────────────┘
                │ (C-contiguous float64)
┌───────────────▼──────────────┐
│ C++ layer (kernel)            │
│  - assignment + update loops  │
│  - minimal allocations        │
│  File: kmeans_cpp.cpp         │
└──────────────────────────────┘
```

Key boundary decisions:

- The Python API passes a single dense `float64` array `X` into C++.
- The C++ implementation releases the Python GIL during the hot loop.
- Both implementations expose comparable outputs (labels, centers, inertia, iteration count).

## Performance snapshot

The benchmark preset used by the web app is:

- `k=25`
- `n_features=10`
- `points_per_cluster=650` (so `N = 25 * 650 = 16250`)
- `max_iter=35`, `n_init=1`, `init=kmeans++`, `random_state=7`

Example wall-time results (single run):

| Implementation | Dataset | Runtime |
| --- | --- | ---: |
| Python | `N=16250, d=10, k=25` | ~19.87 s |
| C++ (pybind11) | `N=16250, d=10, k=25` | ~0.064 s |

Numbers vary by machine and compiler flags.

## Reproducibility

- The benchmark dataset generation uses a fixed `random_state`.
- Each implementation is deterministic given a fixed seed and fixed inputs.
- Due to floating-point and tie-breaking differences, Python and C++ may converge to slightly different local minima; this project focuses on timing and comparable convergence behavior.

## Profiling evidence

This repository includes lightweight profiling scripts:

- `profile_python.py`: runs the Python implementation under `cProfile` and prints a summary.
- `profile_cpp.py`: runs the C++ implementation and prints the timing breakdown reported by the extension.

Run them locally:

```bash
python profile_python.py
python profile_cpp.py
```

Typical outcome:

- On the benchmark preset on one run (`python profile_python.py`, total ~13.42s), `cProfile` showed the assignment step dominating runtime:
  - `kmeans.py:_assign_naive` ~11.98s self time (~89% of total), ~13.03s cumulative (~97% of total)
  - `numpy.dot` inside the assignment loop ~1.04s self time (~8% of total)
  - `kmeans.py:_update_naive` ~0.38s self time (~3% of total)

This motivates moving the assignment/update kernel into C++ while keeping Python for orchestration.

## Hosted app

- Live app (Hugging Face Spaces): https://huggingface.co/spaces/Zivana/kmeans-pybind11-demo
- Canonical repository + documentation (GitHub): https://github.com/adriendomoison/kmeans-pybind11-demo

## Overview

- **`pybind11` extension**: binding a C++ implementation as an importable Python module.
- **Performance**: the C++ implementation releases the Python GIL during the tight loop.
- **API**: a minimal FastAPI backend + static frontend for side-by-side comparison.

## Quickstart (local)

### 1) Install + build the extension

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2) Run the web app

```bash
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Open:

- http://127.0.0.1:8000

## Repository structure

- `web_app.py`: FastAPI service used for the deployed app and benchmark preset generation.
- `static/`: minimal UI that triggers Python and C++ runs in parallel and visualizes results.
- `kmeans.py`: reference Python implementation of k-means (baseline + profiling target).
- `kmeans_cpp.cpp`: C++ k-means kernel (assignment + update loops) exposed via `pybind11`.
- `setup.py`, `pyproject.toml`: packaging/build configuration for the extension.
- `profile_python.py`, `profile_cpp.py`: reproducible profiling/timing entrypoints.
- `run_kmeans.py`: optional CLI runner for local timing.

## Running experiments

Reproduce profiling evidence:

```bash
python profile_python.py
python profile_cpp.py
```

Run the interactive UI locally:

```bash
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

## Notes

- The benchmark preset uses `algorithm="naive"` for the Python implementation to make interpreter overhead in the assignment loop visible in profiling.
- The C++ implementation mirrors the same algorithmic steps but executes the hot loops in compiled code and releases the GIL.

## Legacy / reference components
 
This repository also contains DBSCAN code (`dbscan.py`, `run_dbscan.py`, and optional `dbscan_cpp.cpp`) kept as a secondary reference for clustering algorithms and extension-module structure. It is not used by the k-means implementation.
