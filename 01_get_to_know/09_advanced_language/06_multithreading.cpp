// 09_advanced_language/06_multithreading.cpp
// Multiple threads: several functions making progress at once.
//
// WHAT YOU LEARN:
//   - std::thread(f, a, b) runs the function f(a, b) on a new thread.
//   - You MUST call join() so the main thread waits for each worker,
//     otherwise the program may end mid-work.
//   - Writing to a SHARED variable from several threads is a data race;
//     std::atomic<int> makes increments correct without locks.
//   - Each worker writes into its OWN slot -> no race, simple rule.
//
// EXPECTED OUTPUT:
//   worker 0 wrote slot 0 = 0
//   worker 1 wrote slot 1 = 1
//   worker 2 wrote slot 2 = 4
//   worker 3 wrote slot 3 = 9
//   shared atomic counter  = 400000 (4 workers x 100000 increments)
//   all workers joined; main continues

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

// Each worker writes its own slot, then bumps the shared atomic counter.
void worker(std::vector<int>& slots, int idx, std::atomic<int>& counter) {
    slots[idx] = idx * idx;              // own slot: no race
    for (int i = 0; i < 100000; ++i) {
        ++counter;                       // atomic: threads safe to share
    }
}

int main() {
    const int kWorkers = 4;
    std::vector<int> slots(kWorkers);
    std::atomic<int> counter{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < kWorkers; ++i) {
        threads.emplace_back(worker, std::ref(slots), i,
                             std::ref(counter));
    }
    for (auto& t : threads) {
        t.join();                        // wait for every worker
    }

    for (int i = 0; i < kWorkers; ++i) {
        std::cout << "worker " << i << " wrote slot " << i << " = "
                  << slots[i] << "\n";
    }
    std::cout << "shared atomic counter  = " << counter
              << " (4 workers x 100000 increments)\n";
    std::cout << "all workers joined; main continues\n";
    return 0;
}