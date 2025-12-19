#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

struct Timings {
    double validate_s = 0.0;
    double init_s = 0.0;
    double assign_s = 0.0;
    double update_s = 0.0;
    double total_s = 0.0;
};

static inline double seconds_since(std::chrono::high_resolution_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
}

static inline double dist2_point_center(const double* X, int64_t d, int64_t i, const double* centers, int64_t k) {
    const double* xi = X + i * d;
    const double* ck = centers + k * d;
    double s = 0.0;
    for (int64_t j = 0; j < d; ++j) {
        const double diff = xi[j] - ck[j];
        s += diff * diff;
    }
    return s;
}

static std::vector<int64_t> sample_without_replacement(std::mt19937_64& gen, int64_t n, int64_t k) {
    std::vector<int64_t> idx(n);
    for (int64_t i = 0; i < n; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), gen);
    idx.resize(static_cast<size_t>(k));
    return idx;
}

static void init_random(const double* X, int64_t n, int64_t d, int64_t k, std::mt19937_64& gen, std::vector<double>& centers) {
    centers.assign(static_cast<size_t>(k * d), 0.0);
    auto idx = sample_without_replacement(gen, n, k);
    for (int64_t c = 0; c < k; ++c) {
        const double* src = X + idx[static_cast<size_t>(c)] * d;
        std::copy(src, src + d, centers.begin() + c * d);
    }
}

static void init_kmeanspp(const double* X, int64_t n, int64_t d, int64_t k, std::mt19937_64& gen, std::vector<double>& centers) {
    centers.assign(static_cast<size_t>(k * d), 0.0);

    std::uniform_int_distribution<int64_t> uni(0, n - 1);
    const int64_t first = uni(gen);
    std::copy(X + first * d, X + first * d + d, centers.begin());

    std::vector<double> closest_dist2(static_cast<size_t>(n), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        closest_dist2[static_cast<size_t>(i)] = dist2_point_center(X, d, i, centers.data(), 0);
    }

    for (int64_t c = 1; c < k; ++c) {
        double sum = 0.0;
        for (double v : closest_dist2) sum += v;

        int64_t next_idx = 0;
        if (sum <= 0.0 || !std::isfinite(sum)) {
            next_idx = uni(gen);
        } else {
            std::uniform_real_distribution<double> ur(0.0, sum);
            const double r = ur(gen);
            double acc = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                acc += closest_dist2[static_cast<size_t>(i)];
                if (acc >= r) {
                    next_idx = i;
                    break;
                }
            }
        }

        std::copy(X + next_idx * d, X + next_idx * d + d, centers.begin() + c * d);

        for (int64_t i = 0; i < n; ++i) {
            const double d2 = dist2_point_center(X, d, i, centers.data(), c);
            if (d2 < closest_dist2[static_cast<size_t>(i)]) {
                closest_dist2[static_cast<size_t>(i)] = d2;
            }
        }
    }
}

static py::dict kmeans_cpp(
    py::array X_in,
    int n_clusters,
    int max_iter,
    double tol,
    const std::string& init,
    int n_init,
    py::object random_state,
    bool profile
) {
    auto t_total_start = std::chrono::high_resolution_clock::now();

    Timings timings;
    auto t_validate_start = std::chrono::high_resolution_clock::now();

    if (n_clusters <= 0) throw std::invalid_argument("n_clusters must be >= 1");
    if (max_iter <= 0) throw std::invalid_argument("max_iter must be >= 1");
    if (tol < 0.0) throw std::invalid_argument("tol must be >= 0");
    if (n_init <= 0) throw std::invalid_argument("n_init must be >= 1");

    py::array_t<double, py::array::c_style | py::array::forcecast> X(X_in);
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be a 2D array of shape (n_samples, n_features)");
    }

    const int64_t n = X.shape(0);
    const int64_t d = X.shape(1);
    if (n <= 0) throw std::invalid_argument("X must have at least one sample");
    if (static_cast<int64_t>(n_clusters) > n) throw std::invalid_argument("n_clusters must be <= n_samples");

    py::buffer_info buf = X.request();
    const double* Xptr = static_cast<const double*>(buf.ptr);

    if (profile) {
        timings.validate_s = seconds_since(t_validate_start);
    }

    uint64_t seed;
    if (random_state.is_none()) {
        std::random_device rd;
        seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    } else {
        seed = static_cast<uint64_t>(py::cast<int64_t>(random_state));
    }
    std::mt19937_64 gen(seed);

    const int64_t k = static_cast<int64_t>(n_clusters);

    double best_inertia = std::numeric_limits<double>::infinity();
    int best_n_iter = 0;
    std::vector<int32_t> best_labels(static_cast<size_t>(n), 0);
    std::vector<double> best_centers(static_cast<size_t>(k * d), 0.0);

    auto t_init_start = std::chrono::high_resolution_clock::now();

    std::vector<int32_t> labels(static_cast<size_t>(n), 0);
    std::vector<double> centers;
    std::vector<double> new_centers(static_cast<size_t>(k * d), 0.0);
    std::vector<int64_t> counts(static_cast<size_t>(k), 0);

    {
        py::gil_scoped_release release;

        auto init_once = [&]() {
            if (init == "random") {
                init_random(Xptr, n, d, k, gen, centers);
            } else if (init == "kmeans++") {
                init_kmeanspp(Xptr, n, d, k, gen, centers);
            } else {
                throw std::invalid_argument("init must be 'kmeans++' or 'random'");
            }
        };

        for (int run = 0; run < n_init; ++run) {
            init_once();

            double inertia = std::numeric_limits<double>::infinity();

            for (int it = 0; it < max_iter; ++it) {
                auto t_assign_start = std::chrono::high_resolution_clock::now();
                inertia = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    double best_d2 = std::numeric_limits<double>::infinity();
                    int32_t best_k = 0;
                    for (int64_t c = 0; c < k; ++c) {
                        const double d2 = dist2_point_center(Xptr, d, i, centers.data(), c);
                        if (d2 < best_d2) {
                            best_d2 = d2;
                            best_k = static_cast<int32_t>(c);
                        }
                    }
                    labels[static_cast<size_t>(i)] = best_k;
                    inertia += best_d2;
                }
                if (profile) timings.assign_s += seconds_since(t_assign_start);

                auto t_update_start = std::chrono::high_resolution_clock::now();
                std::fill(new_centers.begin(), new_centers.end(), 0.0);
                std::fill(counts.begin(), counts.end(), 0);

                for (int64_t i = 0; i < n; ++i) {
                    const int64_t c = static_cast<int64_t>(labels[static_cast<size_t>(i)]);
                    counts[static_cast<size_t>(c)] += 1;
                    const double* xi = Xptr + i * d;
                    double* acc = new_centers.data() + c * d;
                    for (int64_t j = 0; j < d; ++j) {
                        acc[j] += xi[j];
                    }
                }

                std::uniform_int_distribution<int64_t> uni(0, n - 1);
                for (int64_t c = 0; c < k; ++c) {
                    double* dst = new_centers.data() + c * d;
                    const int64_t cnt = counts[static_cast<size_t>(c)];
                    if (cnt == 0) {
                        const int64_t idx = uni(gen);
                        std::copy(Xptr + idx * d, Xptr + idx * d + d, dst);
                    } else {
                        const double inv = 1.0 / static_cast<double>(cnt);
                        for (int64_t j = 0; j < d; ++j) {
                            dst[j] *= inv;
                        }
                    }
                }
                if (profile) timings.update_s += seconds_since(t_update_start);

                double shift2 = 0.0;
                for (int64_t c = 0; c < k; ++c) {
                    const double* oldc = centers.data() + c * d;
                    const double* newc = new_centers.data() + c * d;
                    for (int64_t j = 0; j < d; ++j) {
                        const double diff = newc[j] - oldc[j];
                        shift2 += diff * diff;
                    }
                }

                centers.swap(new_centers);

                if (std::sqrt(shift2) <= tol) {
                    best_n_iter = it + 1;
                    break;
                }
                best_n_iter = it + 1;
            }

            if (inertia < best_inertia) {
                best_inertia = inertia;
                best_labels = labels;
                best_centers = centers;
            }
        }
    }

    if (profile) {
        timings.init_s = seconds_since(t_init_start);
        timings.total_s = seconds_since(t_total_start);
    }

    py::array_t<int32_t> labels_out({n});
    py::array_t<double> centers_out({k, d});

    auto labels_u = labels_out.mutable_unchecked<1>();
    for (int64_t i = 0; i < n; ++i) {
        labels_u(i) = best_labels[static_cast<size_t>(i)];
    }

    auto centers_u = centers_out.mutable_unchecked<2>();
    for (int64_t c = 0; c < k; ++c) {
        for (int64_t j = 0; j < d; ++j) {
            centers_u(c, j) = best_centers[static_cast<size_t>(c * d + j)];
        }
    }

    py::object timing_obj = py::none();
    if (profile) {
        py::dict t;
        t["validate_s"] = timings.validate_s;
        t["init_s"] = timings.init_s;
        t["assign_s"] = timings.assign_s;
        t["update_s"] = timings.update_s;
        t["total_s"] = timings.total_s;
        timing_obj = std::move(t);
    }

    py::dict out;
    out["labels"] = std::move(labels_out);
    out["centers"] = std::move(centers_out);
    out["inertia"] = py::float_(best_inertia);
    out["n_iter"] = py::int_(best_n_iter);
    out["timing"] = std::move(timing_obj);
    return out;
}

PYBIND11_MODULE(kmeans_cpp, m) {
    m.def(
        "kmeans",
        &kmeans_cpp,
        py::arg("X"),
        py::arg("n_clusters"),
        py::arg("max_iter") = 300,
        py::arg("tol") = 1e-4,
        py::arg("init") = std::string("kmeans++"),
        py::arg("n_init") = 1,
        py::arg("random_state") = py::none(),
        py::arg("profile") = false
    );
}
