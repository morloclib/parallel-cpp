#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
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
    std::atomic<bool> stop{false};
    std::exception_ptr first;
    std::mutex first_mtx;

    auto worker = [&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            const std::size_t i = cursor.fetch_add(1, std::memory_order_relaxed);
            if (i >= nspans) return;
            try {
                body(i);
            } catch (...) {
                stop.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lk(first_mtx);
                if (!first) first = std::current_exception();
                return;
            }
        }
    };

    // A joinable std::thread destroyed during unwinding calls std::terminate,
    // so a failure to start a thread must still join the ones already running.
    std::vector<std::thread> threads;
    threads.reserve(nthreads - 1);
    try {
        for (std::size_t t = 0; t + 1 < nthreads; ++t) threads.emplace_back(worker);
    } catch (...) {
        stop.store(true, std::memory_order_relaxed);
        for (auto& t : threads) t.join();
        throw;
    }
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

// Streams
// -------
//
// A stream stage is a pipeline. The calling thread pulls batches and sinks
// results -- the source and sink are morloc closures bound to this thread's
// call -- while a set of worker threads started once per stage maps work
// units. `inflight` counts units dispatched and not yet delivered, so it bounds
// the reorder buffer as well as the queue.
//
// Every exit path, including an exception from the pull, the sink or a
// combine, stops and joins the workers before leaving: a joinable std::thread
// destroyed during unwinding calls std::terminate.

namespace mlcpar {

inline std::size_t inflight_limit(const ParOpts& opts, std::size_t w) {
    return std::max<std::size_t>(1, positive_or(opts.inflight, 2 * w));
}

template <class A, class Out>
class StreamWorkers {
public:
    struct Unit {
        std::size_t seq;
        std::shared_ptr<const std::vector<A>> batch;
        std::size_t lo;
        std::size_t hi;
    };
    struct Done {
        std::size_t seq;
        Out out;
        std::exception_ptr error;
    };

    template <class Compute>
    StreamWorkers(std::size_t n, Compute compute) {
        threads_.reserve(n);
        try {
            for (std::size_t t = 0; t < n; ++t) {
                threads_.emplace_back([this, compute]() { run(compute); });
            }
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~StreamWorkers() { shutdown(); }

    void submit(Unit u) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            work_.push_back(std::move(u));
        }
        work_cv_.notify_one();
    }

    // Block until at least one unit has completed, and take every completed one.
    std::vector<Done> wait_done() {
        std::unique_lock<std::mutex> lk(mtx_);
        done_cv_.wait(lk, [this] { return !done_.empty(); });
        std::vector<Done> out;
        out.swap(done_);
        return out;
    }

    // Queued units not yet started are skipped once this is set.
    void abandon() { stop_.store(true, std::memory_order_relaxed); }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            closing_ = true;
        }
        stop_.store(true, std::memory_order_relaxed);
        work_cv_.notify_all();
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

private:
    template <class Compute>
    void run(Compute compute) {
        for (;;) {
            Unit u;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                work_cv_.wait(lk, [this] { return closing_ || !work_.empty(); });
                if (work_.empty()) return;
                u = std::move(work_.front());
                work_.pop_front();
            }
            Done d{u.seq, Out{}, nullptr};
            if (!stop_.load(std::memory_order_relaxed)) {
                try {
                    d.out = compute(*u.batch, u.lo, u.hi);
                } catch (...) {
                    d.error = std::current_exception();
                }
            }
            {
                std::lock_guard<std::mutex> lk(mtx_);
                done_.push_back(std::move(d));
            }
            done_cv_.notify_one();
        }
    }

    std::vector<std::thread> threads_;
    std::mutex mtx_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::deque<Unit> work_;
    std::vector<Done> done_;
    bool closing_ = false;
    std::atomic<bool> stop_{false};
};

// Run a stream stage. `compute(batch, lo, hi)` maps one unit; `deliver`
// receives each non-empty unit result, in input order when `ordered`.
// Returns {true, ""} at end of stream or {false, msg} after a read failure,
// once every dispatched unit has been delivered. A failure in a unit or in
// `deliver` propagates.
template <class A, class Out, class Pull, class Compute, class Deliver>
std::tuple<bool, std::string> pipeline(const ParOpts& opts, Pull& pull, Compute compute,
                                       Deliver deliver, bool ordered) {
    const std::size_t w = worker_count(opts);

    if (w <= 1) {
        for (;;) {
            auto [ok, msg, xs] = pull();
            if (!ok) return {false, msg};
            if (xs.empty()) return {true, std::string()};
            for (const auto& [lo, hi] : ranges(xs.size(), opts)) {
                Out ys = compute(xs, lo, hi);
                if (!ys.empty()) deliver(std::move(ys));
            }
        }
    }

    using Workers = StreamWorkers<A, Out>;
    Workers workers(w, compute);
    const std::size_t limit = inflight_limit(opts, w);

    std::shared_ptr<const std::vector<A>> batch;
    std::vector<Span> spans;
    std::size_t span_i = 0;
    bool reading = true;
    std::tuple<bool, std::string> status{true, std::string()};
    std::exception_ptr failure;
    std::size_t next_seq = 0, next_out = 0, running = 0, undelivered = 0;
    std::map<std::size_t, Out> held;

    for (;;) {
        while (reading && !failure && undelivered < limit) {
            if (span_i == spans.size()) {
                auto [ok, msg, xs] = pull();
                if (!ok) {
                    reading = false;
                    status = {false, msg};
                    break;
                }
                if (xs.empty()) {
                    reading = false;
                    break;
                }
                batch = std::make_shared<const std::vector<A>>(std::move(xs));
                spans = ranges(batch->size(), opts);
                span_i = 0;
            }
            const auto [lo, hi] = spans[span_i++];
            workers.submit({next_seq++, batch, lo, hi});
            ++running;
            ++undelivered;
        }

        if (running == 0) break;

        for (auto& d : workers.wait_done()) {
            --running;
            if (d.error) {
                if (!failure) failure = d.error;
                workers.abandon();
                continue;
            }
            if (failure) continue;
            if (ordered) {
                held.emplace(d.seq, std::move(d.out));
                for (auto it = held.find(next_out); it != held.end(); it = held.find(next_out)) {
                    Out ys = std::move(it->second);
                    held.erase(it);
                    ++next_out;
                    --undelivered;
                    if (!ys.empty()) deliver(std::move(ys));
                }
            } else {
                --undelivered;
                if (!d.out.empty()) deliver(std::move(d.out));
            }
        }
    }

    workers.shutdown();
    if (failure) std::rethrow_exception(failure);
    return status;
}

template <class Pull>
using pulled_batch_t = std::decay_t<decltype(std::get<2>(std::declval<Pull&>()()))>;

}  // namespace mlcpar

template <class F, class Pull, class Sink>
std::tuple<bool, std::string> psconcatMapNative(const ParOpts& opts, F f, Pull pull, Sink sink) {
    using Batch = mlcpar::pulled_batch_t<Pull>;
    using A = typename Batch::value_type;
    using Out = std::decay_t<std::invoke_result_t<F, const A&>>;
    auto compute = [f](const std::vector<A>& xs, std::size_t lo, std::size_t hi) {
        Out out;
        for (std::size_t j = lo; j < hi; ++j) {
            Out piece = f(xs[j]);
            out.insert(out.end(), std::make_move_iterator(piece.begin()),
                       std::make_move_iterator(piece.end()));
        }
        return out;
    };
    auto deliver = [&sink](Out ys) { sink(ys); };
    return mlcpar::pipeline<A, Out>(opts, pull, compute, deliver, opts.order == ParOrder::InputOrder);
}

template <class F, class Pull, class Sink>
std::tuple<bool, std::string> psmapNative(const ParOpts& opts, F f, Pull pull, Sink sink) {
    using Batch = mlcpar::pulled_batch_t<Pull>;
    using A = typename Batch::value_type;
    using Out = std::vector<std::decay_t<std::invoke_result_t<F, const A&>>>;
    auto compute = [f](const std::vector<A>& xs, std::size_t lo, std::size_t hi) {
        Out out;
        out.reserve(hi - lo);
        for (std::size_t j = lo; j < hi; ++j) out.emplace_back(f(xs[j]));
        return out;
    };
    auto deliver = [&sink](Out ys) { sink(ys); };
    return mlcpar::pipeline<A, Out>(opts, pull, compute, deliver, opts.order == ParOrder::InputOrder);
}

template <class P, class Pull, class Sink>
std::tuple<bool, std::string> psfilterNative(const ParOpts& opts, P pred, Pull pull, Sink sink) {
    using Batch = mlcpar::pulled_batch_t<Pull>;
    using A = typename Batch::value_type;
    using Out = std::vector<A>;
    auto compute = [pred](const std::vector<A>& xs, std::size_t lo, std::size_t hi) {
        Out out;
        for (std::size_t j = lo; j < hi; ++j) {
            if (pred(xs[j])) out.push_back(xs[j]);
        }
        return out;
    };
    auto deliver = [&sink](Out ys) { sink(ys); };
    return mlcpar::pipeline<A, Out>(opts, pull, compute, deliver, opts.order == ParOrder::InputOrder);
}

// The fold always consumes results in input order, whatever `order` asks: a
// fixed fold order is what makes the answer schedule-independent.
template <class C, class Acc, class F, class Pull>
std::tuple<bool, std::string, Acc> psfoldNative(const ParOpts& opts, C combine, Acc identity, F f,
                                                 Pull pull) {
    using Batch = mlcpar::pulled_batch_t<Pull>;
    using A = typename Batch::value_type;
    using B = std::decay_t<std::invoke_result_t<F, const A&>>;
    using Out = std::vector<B>;
    auto compute = [f](const std::vector<A>& xs, std::size_t lo, std::size_t hi) {
        Out out;
        out.reserve(hi - lo);
        for (std::size_t j = lo; j < hi; ++j) out.emplace_back(f(xs[j]));
        return out;
    };
    Acc acc = std::move(identity);
    auto deliver = [&](Out ys) {
        for (auto& y : ys) acc = combine(acc, y);
    };
    auto [ok, msg] = mlcpar::pipeline<A, Out>(opts, pull, compute, deliver, true);
    return {ok, msg, std::move(acc)};
}
