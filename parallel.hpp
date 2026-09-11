#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// User-mapped types, matching the declarations in parallel/main.loc. Field and
// tag order are the wire contract: a record is positional and a constructor's
// tag is its position, so reordering either here silently misreads values.
enum class ParChunking : uint8_t { EvenChunks = 0, ShrinkingChunks = 1, FixedChunks = 2 };
enum class ParOrder : uint8_t { InputOrder = 0, ArrivalOrder = 1 };

struct ParOpts {
    std::optional<int> workers;
    std::optional<int> chunkSize;
    ParChunking chunking;
    ParOrder order;
    std::optional<int> inflight;
};

namespace mlcpar {

using Span = std::pair<std::size_t, std::size_t>;

inline std::size_t positive_or(const std::optional<int>& v, std::size_t fallback) {
    if (v.has_value() && *v > 0) return static_cast<std::size_t>(*v);
    return fallback;
}

inline std::size_t worker_count(const ParOpts& opts) {
    unsigned hw = std::thread::hardware_concurrency();
    return positive_or(opts.workers, hw == 0 ? 1 : static_cast<std::size_t>(hw));
}

// Split [0, n) into work units per the requested schedule. Identical in shape
// to the Python backend's `_ranges`, and deliberately so: the schedule is part
// of the module's meaning, not of any one language's implementation.
inline std::vector<Span> ranges(std::size_t n, const ParOpts& opts) {
    std::vector<Span> out;
    if (n == 0) return out;
    const std::size_t w = std::max<std::size_t>(1, worker_count(opts));

    if (opts.chunking == ParChunking::EvenChunks) {
        const std::size_t size = (n + w - 1) / w;
        for (std::size_t i = 0; i < n; i += size) out.emplace_back(i, std::min(i + size, n));
        return out;
    }

    if (opts.chunking == ParChunking::FixedChunks) {
        const std::size_t size = std::max<std::size_t>(1, positive_or(opts.chunkSize, 1));
        for (std::size_t i = 0; i < n; i += size) out.emplace_back(i, std::min(i + size, n));
        return out;
    }

    // Shrinking (factoring): each round deals w units of half the remaining
    // work, so early units are large and the tail is single elements.
    std::size_t lo = 0;
    while (lo < n) {
        const std::size_t size = std::max<std::size_t>(1, (n - lo) / (2 * w));
        for (std::size_t k = 0; k < w && lo < n; ++k) {
            const std::size_t hi = std::min(lo + size, n);
            out.emplace_back(lo, hi);
            lo = hi;
        }
    }
    return out;
}

// Run `body(span_index)` over every span on a thread pool that pulls from a
// shared cursor -- which is what makes a shrinking schedule dynamic rather
// than merely uneven.
//
// Every worker body is wrapped: an exception escaping a std::thread's function
// calls std::terminate, which would take the whole pool down and report a
// crash instead of the user's error. The first failure is rethrown on the
// calling thread once the workers have joined.
template <class Body>
inline void run_spans(const ParOpts& opts, std::size_t nspans, Body body) {
    if (nspans == 0) return;
    if (nspans == 1) { body(0); return; }

    const std::size_t nthreads = std::min(std::max<std::size_t>(1, worker_count(opts)), nspans);
    std::atomic<std::size_t> cursor{0};
    std::exception_ptr first;
    std::mutex first_mtx;

    auto worker = [&]() {
        for (;;) {
            const std::size_t i = cursor.fetch_add(1, std::memory_order_relaxed);
            if (i >= nspans) return;
            try {
                body(i);
            } catch (...) {
                std::lock_guard<std::mutex> lk(first_mtx);
                if (!first) first = std::current_exception();
                return;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(nthreads - 1);
    for (std::size_t t = 0; t + 1 < nthreads; ++t) threads.emplace_back(worker);
    worker();
    for (auto& t : threads) t.join();
    if (first) std::rethrow_exception(first);
}

template <class T>
inline std::vector<T> flatten(std::vector<std::vector<T>>& parts) {
    std::size_t total = 0;
    for (const auto& p : parts) total += p.size();
    std::vector<T> out;
    out.reserve(total);
    for (auto& p : parts) {
        out.insert(out.end(), std::make_move_iterator(p.begin()), std::make_move_iterator(p.end()));
        std::vector<T>().swap(p);
    }
    return out;
}

}  // namespace mlcpar

template <class F, class A>
auto pmapWith(const ParOpts& opts, F f, const std::vector<A>& xs)
    -> std::vector<std::decay_t<std::invoke_result_t<F, const A&>>> {
    using B = std::decay_t<std::invoke_result_t<F, const A&>>;
    const auto spans = mlcpar::ranges(xs.size(), opts);
    std::vector<std::vector<B>> parts(spans.size());
    mlcpar::run_spans(opts, spans.size(), [&](std::size_t i) {
        const auto [lo, hi] = spans[i];
        parts[i].reserve(hi - lo);
        for (std::size_t j = lo; j < hi; ++j) parts[i].emplace_back(f(xs[j]));
    });
    return mlcpar::flatten(parts);
}

template <class F, class A>
auto pconcatMapWith(const ParOpts& opts, F f, const std::vector<A>& xs)
    -> std::decay_t<std::invoke_result_t<F, const A&>> {
    using Out = std::decay_t<std::invoke_result_t<F, const A&>>;
    const auto spans = mlcpar::ranges(xs.size(), opts);
    std::vector<Out> parts(spans.size());
    mlcpar::run_spans(opts, spans.size(), [&](std::size_t i) {
        const auto [lo, hi] = spans[i];
        for (std::size_t j = lo; j < hi; ++j) {
            Out piece = f(xs[j]);
            parts[i].insert(parts[i].end(),
                            std::make_move_iterator(piece.begin()),
                            std::make_move_iterator(piece.end()));
        }
    });
    return mlcpar::flatten(parts);
}

template <class P, class A>
inline std::vector<A> pfilterWith(const ParOpts& opts, P pred, const std::vector<A>& xs) {
    const auto spans = mlcpar::ranges(xs.size(), opts);
    // The predicate runs in parallel; the compaction is a sequential pass, so
    // the surviving elements keep their input order without a merge step.
    std::vector<char> keep(xs.size(), 0);
    mlcpar::run_spans(opts, spans.size(), [&](std::size_t i) {
        const auto [lo, hi] = spans[i];
        for (std::size_t j = lo; j < hi; ++j) keep[j] = pred(xs[j]) ? 1 : 0;
    });
    std::vector<A> out;
    for (std::size_t j = 0; j < xs.size(); ++j) {
        if (keep[j]) out.push_back(xs[j]);
    }
    return out;
}
