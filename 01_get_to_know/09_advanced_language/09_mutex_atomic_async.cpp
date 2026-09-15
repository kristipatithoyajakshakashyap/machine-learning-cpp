// 09_advanced_language/09_mutex_atomic_async.cpp
// Four ways to share work between threads: broken, mutex, atomic, async.
//
// WHAT YOU LEARN:
//   - Plain `++counter` from several threads is a data race: increments can
//     be lost, and the result differs from run to run (nondeterministic).
//   - std::mutex + std::lock_guard serialises the critical section; the
//     guard unlocks automatically when it goes out of scope.
//   - std::atomic<int> makes the increment itself indivisible: no lock.
//   - std::async(std::launch::async, f) runs f on another thread and hands
//     back a std::future; .get() waits and returns the value.
//   - Every thread is joined (or its future retrieved) before printing.
//
// EXPECTED OUTPUT:
//   unsynchronized counter = <value <= 8000000> of 8000000 (nondeterministic!)
//   mutex-protected counter = 400000
//   atomic counter          = 400000
//   async: sum(1..500000) + sum(500001..1000000) = 500000500000
//   all workers finished

#include <atomic>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

constexpr int kWorkers = 4;
constexpr int kIncrements = 100000;
constexpr int kRaceIncrements = 2000000;  // long enough for the threads to overlap

// Run `body` on kWorkers threads and join them all. A "starting gun" flag
// makes every worker begin at the same moment (thread start-up is slow, so
// without it the workers would often run one after another and hide races).
template <typename F>
void run_workers(F body) {
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    for (int i = 0; i < kWorkers; ++i) {
        threads.emplace_back([&go, body] {
            while (!go.load()) { /* spin until all workers exist */ }
            body();
        });
    }
    go.store(true);
    for (auto& t : threads) t.join();
}

long long sum_range(long long lo, long long hi) {  // inclusive
    long long s = 0;
    for (long long v = lo; v <= hi; ++v) s += v;
    return s;
}

int main() {
    // 1. Data race: DO NOT copy this pattern. Shown only to see the problem.
    //    `volatile` only stops the optimizer from folding the loop into one
    //    store per thread; it does NOT make the increment thread-safe.
    volatile int racy = 0;
    run_workers([&racy] {
        for (int i = 0; i < kRaceIncrements; ++i) racy = racy + 1;  // load, add, store: unprotected
    });
    std::cout << "unsynchronized counter = " << racy << " of "
              << kWorkers * kRaceIncrements
              << " (nondeterministic! usually less: lost increments)\n";

    // 2. Mutex: only one thread inside the braces at a time.
    int guarded = 0;
    std::mutex m;
    run_workers([&guarded, &m] {
        for (int i = 0; i < kIncrements; ++i) {
            std::lock_guard<std::mutex> lock(m);
            ++guarded;
        }
    });
    std::cout << "mutex-protected counter = " << guarded << '\n';

    // 3. Atomic: the hardware does the increment as one step.
    std::atomic<int> atomic_counter{0};
    run_workers([&atomic_counter] {
        for (int i = 0; i < kIncrements; ++i) ++atomic_counter;
    });
    std::cout << "atomic counter          = " << atomic_counter.load() << '\n';

    // 4. async/future: two halves computed in parallel, results returned.
    std::future<long long> lo = std::async(std::launch::async, sum_range, 1LL, 500000LL);
    std::future<long long> hi = std::async(std::launch::async, sum_range, 500001LL, 1000000LL);
    const long long total = lo.get() + hi.get();  // get() waits for each thread
    std::cout << "async: sum(1..500000) + sum(500001..1000000) = " << total << '\n';

    std::cout << "all workers finished\n";
    return 0;
}
