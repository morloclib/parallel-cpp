#pragma once

// Synthetic stream source and sink for the shared `parallel` test suite. A
// source is a list of batch sizes; its elements count up from 0 across
// batches, and a negative size is a read failure at that point. A sink records
// every batch it receives. The pool serves calls on several threads, so the
// registries are locked.

#include <mutex>
#include <tuple>
#include <vector>

#include "mlccpptypes/prelude.hpp"

namespace mlcpartest {

struct Source {
    std::vector<int> sizes;
    std::size_t i = 0;
    int start = 0;
};

inline std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

inline std::vector<Source>& sources() {
    static std::vector<Source> s;
    return s;
}

inline std::vector<std::vector<std::vector<int>>>& sinks() {
    static std::vector<std::vector<std::vector<int>>> s;
    return s;
}

}  // namespace mlcpartest

inline int synthOpen(const std::vector<int>& sizes) {
    std::lock_guard<std::mutex> lk(mlcpartest::registry_mutex());
    mlcpartest::sources().push_back({sizes, 0, 0});
    return static_cast<int>(mlcpartest::sources().size()) - 1;
}

inline std::tuple<bool, std::vector<int>> synthNext(int k) {
    std::lock_guard<std::mutex> lk(mlcpartest::registry_mutex());
    auto& s = mlcpartest::sources().at(k);
    if (s.i >= s.sizes.size()) return {true, {}};
    const int n = s.sizes[s.i++];
    if (n < 0) return {false, {}};
    std::vector<int> xs(n);
    for (int j = 0; j < n; ++j) xs[j] = s.start + j;
    s.start += n;
    return {true, xs};
}

inline int sinkOpen() {
    std::lock_guard<std::mutex> lk(mlcpartest::registry_mutex());
    mlcpartest::sinks().emplace_back();
    return static_cast<int>(mlcpartest::sinks().size()) - 1;
}

inline mlc::Unit sinkPut(int k, const std::vector<int>& xs) {
    std::lock_guard<std::mutex> lk(mlcpartest::registry_mutex());
    mlcpartest::sinks().at(k).push_back(xs);
    return mlc::Unit{};
}

inline std::vector<std::vector<int>> sinkBatches(int k) {
    std::lock_guard<std::mutex> lk(mlcpartest::registry_mutex());
    return mlcpartest::sinks().at(k);
}
