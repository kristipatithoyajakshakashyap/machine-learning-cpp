// 07_stl_core/08_random_and_chrono.cpp
// Reproducible random numbers with <random> and wall-clock timing with
// <chrono>: the two headers every experiment and benchmark needs.
//
// WHAT YOU LEARN:
//   - std::mt19937 is the engine (a deterministic bit stream from a seed);
//     a distribution shapes those bits into ints, doubles, or coin flips.
//     Never use rand() % n: it is biased and not reproducible across
//     platforms.
//   - The same seed always gives the same sequence: seed once, log the
//     seed, and any run can be replayed exactly.
//   - std::shuffle takes the engine so the permutation is reproducible too.
//   - std::chrono::steady_clock measures elapsed time (never goes backwards);
//     duration_cast converts to ms/us/ns. A tiny "time this lambda" helper
//     keeps benchmarks honest by returning the result so work is not
//     optimised away.
//
// EXPECTED OUTPUT (the random values are fixed by seed 42 on libstdc++):
//   uniform_int(1..6) x10   : 3 5 6 2 5 5 4 4 1 3
//   uniform_real(0..1) x3   : 0.100 0.459 0.334
//   normal(0, 1) x3         : 0.393 -0.929 0.080
//   bernoulli(0.3) x10      : 0 1 0 0 0 1 1 0 0 1
//   sample mean of 10000 normals ~ 0 : yes
//   shuffle 1..8 (seed 7)   : 3 8 7 5 2 6 4 1
//   same seed, same shuffle : yes
//   different seed differs  : yes
//   sum of (i*i)%7 for i in 1..10^7 = 20000001, took N us   (time varies)
//   1 second = 1000 ms = 1000000 us
//   benchmark(sort 100000 random ints) -> first 3 sorted: 0 10 10 (N us)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// --- simple benchmark helper -----------------------------------------------
// Runs f(), returns {result, elapsed}. Returning the result stops the
// optimiser from deleting the work as "unused".
template <typename F>
auto time_it(F f) {
    const auto start = std::chrono::steady_clock::now();
    auto result = f();
    const auto stop = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return std::pair{result, elapsed};
}

template <typename T>
void print_seq(const char* label, const std::vector<T>& v) {
    std::cout << label << ':';
    for (const T& x : v) std::cout << ' ' << x;
    std::cout << '\n';
}

std::vector<int> shuffled_1_to_8(std::uint32_t seed) {
    std::vector<int> v(8);
    std::iota(v.begin(), v.end(), 1);
    std::mt19937 rng(seed);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

int main() {
    std::cout << std::fixed << std::setprecision(3);

    // --- one engine, several distributions ---------------------------------
    std::mt19937 rng(42);                        // fixed seed: reproducible
    std::uniform_int_distribution<int> die(1, 6);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::bernoulli_distribution coin(0.3);

    std::vector<int> rolls;
    for (int i = 0; i < 10; ++i) rolls.push_back(die(rng));
    print_seq("uniform_int(1..6) x10   ", rolls);

    std::vector<double> reals;
    for (int i = 0; i < 3; ++i) reals.push_back(unit(rng));
    print_seq("uniform_real(0..1) x3   ", reals);

    std::vector<double> normals;
    for (int i = 0; i < 3; ++i) normals.push_back(gauss(rng));
    print_seq("normal(0, 1) x3         ", normals);

    std::vector<int> flips;
    for (int i = 0; i < 10; ++i) flips.push_back(coin(rng) ? 1 : 0);
    print_seq("bernoulli(0.3) x10      ", flips);

    double sum = 0.0;
    for (int i = 0; i < 10000; ++i) sum += gauss(rng);
    std::cout << "sample mean of 10000 normals ~ 0 : "
              << (std::abs(sum / 10000.0) < 0.05 ? "yes" : "no") << '\n';

    // --- reproducible shuffle ----------------------------------------------
    print_seq("shuffle 1..8 (seed 7)   ", shuffled_1_to_8(7));
    std::cout << "same seed, same shuffle : "
              << (shuffled_1_to_8(7) == shuffled_1_to_8(7) ? "yes" : "no") << '\n';
    std::cout << "different seed differs  : "
              << (shuffled_1_to_8(7) != shuffled_1_to_8(8) ? "yes" : "no") << '\n';

    // --- chrono: timing a loop ----------------------------------------------
    // (i*i)%7 keeps the optimiser from replacing the loop with a formula.
    const auto [total, took] = time_it([] {
        std::uint64_t s = 0;
        for (std::uint64_t i = 1; i <= 10'000'000; ++i) s += (i * i) % 7;
        return s;
    });
    std::cout << "sum of (i*i)%7 for i in 1..10^7 = " << total << ", took "
              << took.count() << " us\n";

    const std::chrono::seconds one_second{1};
    std::cout << "1 second = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(one_second).count()
              << " ms = "
              << std::chrono::duration_cast<std::chrono::microseconds>(one_second).count()
              << " us\n";

    // --- benchmark helper on a sort ----------------------------------------
    const auto [sorted, sort_time] = time_it([] {
        std::mt19937 g(1);
        std::uniform_int_distribution<int> d(0, 1'000'000);
        std::vector<int> v(100'000);
        for (int& x : v) x = d(g);
        std::sort(v.begin(), v.end());
        return v;
    });
    std::cout << "benchmark(sort 100000 random ints) -> first 3 sorted: "
              << sorted[0] << ' ' << sorted[1] << ' ' << sorted[2] << " ("
              << sort_time.count() << " us)\n";
    return 0;
}
