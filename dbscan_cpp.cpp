#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

static constexpr int32_t UNCLASSIFIED = -2;
static constexpr int32_t NOISE = -1;

struct Timings {
    double validate_s = 0.0;
    double neighbor_queries_s = 0.0;
    double expand_loop_s = 0.0;
    double total_s = 0.0;
};

static inline double seconds_since(std::chrono::high_resolution_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
}

template <typename T>
static std::vector<int32_t> neighbors_within_eps(const T* X, int64_t n, int64_t d, int64_t index, double eps2) {
    std::vector<int32_t> out;
    out.reserve(64);

    const T* xi = X + index * d;
    for (int64_t j = 0; j < n; ++j) {
        const T* xj = X + j * d;
        double dist2 = 0.0;
        for (int64_t k = 0; k < d; ++k) {
            const double diff = static_cast<double>(xj[k]) - static_cast<double>(xi[k]);
            dist2 += diff * diff;
        }
        if (dist2 <= eps2) {
            out.push_back(static_cast<int32_t>(j));
        }
    }
    return out;
}

static py::dict dbscan_cpp(py::array X_in, double eps, int min_samples, bool profile) {
    auto t_total_start = std::chrono::high_resolution_clock::now();

    Timings timings;
    auto t_validate_start = std::chrono::high_resolution_clock::now();

    if (eps <= 0.0) {
        throw std::invalid_argument("eps must be > 0");
    }
    if (min_samples <= 0) {
        throw std::invalid_argument("min_samples must be >= 1");
    }

    py::array_t<double, py::array::c_style | py::array::forcecast> X(X_in);
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be a 2D array of shape (n_samples, n_features)");
    }

    const int64_t n = X.shape(0);
    const int64_t d = X.shape(1);
    if (n <= 0) {
        throw std::invalid_argument("X must have at least one sample");
    }

    py::buffer_info buf = X.request();
    if (buf.strides.size() != 2) {
        throw std::invalid_argument("X must be a 2D array");
    }

    if (profile) {
        timings.validate_s = seconds_since(t_validate_start);
    }

    std::vector<int32_t> labels(static_cast<size_t>(n), UNCLASSIFIED);
    std::vector<uint8_t> visited(static_cast<size_t>(n), 0);
    std::vector<uint8_t> core_mask(static_cast<size_t>(n), 0);

    const double eps2 = eps * eps;
    int32_t cluster_id = 0;

    auto neighbors_query = [&](int64_t idx) -> std::vector<int32_t> {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> res;
        res = neighbors_within_eps(static_cast<const double*>(buf.ptr), n, d, idx, eps2);
        if (profile) {
            timings.neighbor_queries_s += seconds_since(t0);
        }
        return res;
    };

    for (int64_t i = 0; i < n; ++i) {
        if (visited[static_cast<size_t>(i)]) {
            continue;
        }
        visited[static_cast<size_t>(i)] = 1;

        auto neighbors = neighbors_query(i);
        if (static_cast<int>(neighbors.size()) < min_samples) {
            labels[static_cast<size_t>(i)] = NOISE;
            continue;
        }

        core_mask[static_cast<size_t>(i)] = 1;
        labels[static_cast<size_t>(i)] = cluster_id;

        std::deque<int32_t> seeds;
        seeds.insert(seeds.end(), neighbors.begin(), neighbors.end());

        std::vector<uint8_t> in_seeds(static_cast<size_t>(n), 0);
        for (int32_t j : neighbors) {
            in_seeds[static_cast<size_t>(j)] = 1;
        }
        in_seeds[static_cast<size_t>(i)] = 0;

        for (auto it = seeds.begin(); it != seeds.end();) {
            if (*it == static_cast<int32_t>(i)) {
                it = seeds.erase(it);
            } else {
                ++it;
            }
        }

        auto t_expand_start = std::chrono::high_resolution_clock::now();
        while (!seeds.empty()) {
            const int32_t j = seeds.front();
            seeds.pop_front();
            in_seeds[static_cast<size_t>(j)] = 0;

            if (!visited[static_cast<size_t>(j)]) {
                visited[static_cast<size_t>(j)] = 1;

                auto neighbors_j = neighbors_query(static_cast<int64_t>(j));
                if (static_cast<int>(neighbors_j.size()) >= min_samples) {
                    core_mask[static_cast<size_t>(j)] = 1;
                    for (int32_t k : neighbors_j) {
                        if (labels[static_cast<size_t>(k)] == UNCLASSIFIED) {
                            labels[static_cast<size_t>(k)] = cluster_id;
                        }
                        if (!visited[static_cast<size_t>(k)] && !in_seeds[static_cast<size_t>(k)]) {
                            seeds.push_back(k);
                            in_seeds[static_cast<size_t>(k)] = 1;
                        }
                    }
                }
            }

            if (labels[static_cast<size_t>(j)] == UNCLASSIFIED || labels[static_cast<size_t>(j)] == NOISE) {
                labels[static_cast<size_t>(j)] = cluster_id;
            }
        }
        if (profile) {
            timings.expand_loop_s += seconds_since(t_expand_start);
        }

        cluster_id += 1;
    }

    for (int64_t i = 0; i < n; ++i) {
        if (labels[static_cast<size_t>(i)] == UNCLASSIFIED) {
            labels[static_cast<size_t>(i)] = NOISE;
        }
    }

    if (profile) {
        timings.total_s = seconds_since(t_total_start);
    }

    py::array_t<int32_t> labels_out({n});
    py::array_t<bool> core_out({n});

    auto labels_buf_out = labels_out.mutable_unchecked<1>();
    auto core_buf_out = core_out.mutable_unchecked<1>();

    for (int64_t i = 0; i < n; ++i) {
        labels_buf_out(i) = labels[static_cast<size_t>(i)];
        core_buf_out(i) = core_mask[static_cast<size_t>(i)] != 0;
    }

    py::object timing_obj = py::none();
    if (profile) {
        py::dict t;
        t["validate_s"] = timings.validate_s;
        t["neighbor_queries_s"] = timings.neighbor_queries_s;
        t["expand_loop_s"] = timings.expand_loop_s;
        t["total_s"] = timings.total_s;
        timing_obj = std::move(t);
    }

    py::dict out;
    out["labels"] = std::move(labels_out);
    out["core_sample_mask"] = std::move(core_out);
    out["timing"] = std::move(timing_obj);
    return out;
}

PYBIND11_MODULE(dbscan_cpp, m) {
    m.attr("NOISE") = py::int_(NOISE);
    m.def("dbscan", &dbscan_cpp, py::arg("X"), py::arg("eps"), py::arg("min_samples"), py::arg("profile") = false);
}
