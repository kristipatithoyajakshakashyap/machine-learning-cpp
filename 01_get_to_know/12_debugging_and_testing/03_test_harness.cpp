// 12_debugging_and_testing/03_test_harness.cpp
// A ~40-line test framework: TEST registration, EXPECT_* macros, a runner.
//
// WHAT YOU LEARN:
//   - A static registry (vector of {name, function}) filled by TEST(name)
//     at program start-up lets main() discover every test automatically.
//   - EXPECT_TRUE / EXPECT_EQ / EXPECT_NEAR record a failure with __FILE__
//     and __LINE__ and keep going, so one run reports every broken check.
//   - The process exit code is the number of failed tests != 0, which is
//     what CTest looks at. This is the core of GoogleTest/Catch2 in miniature.
//
// EXPECTED OUTPUT:
//   [ RUN  ] running_stats_basic
//   [ RUN  ] running_stats_single
//   [ RUN  ] running_stats_negative
//   [ RUN  ] running_stats_empty_is_nan
//   4 passed, 0 failed

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// ---------------- mini test framework (~40 lines) ----------------------
namespace mini {
struct Test { std::string name; std::function<void()> body; };
struct Registry {
    std::vector<Test> tests;
    int failures_in_current = 0;
    static Registry& get() { static Registry r; return r; }
};
struct Registrar {
    Registrar(const char* name, std::function<void()> body) {
        Registry::get().tests.push_back({name, std::move(body)});
    }
};
inline void fail(const char* file, int line, const std::string& msg) {
    std::cout << "    FAILED " << file << ":" << line << ": " << msg << '\n';
    ++Registry::get().failures_in_current;
}
inline int run_all() {
    int passed = 0, failed = 0;
    for (const Test& t : Registry::get().tests) {
        std::cout << "[ RUN  ] " << t.name << '\n';
        Registry::get().failures_in_current = 0;
        t.body();
        if (Registry::get().failures_in_current == 0) ++passed; else ++failed;
    }
    std::cout << passed << " passed, " << failed << " failed\n";
    return failed;
}
}  // namespace mini

#define TEST(name) \
    static void name(); \
    static mini::Registrar name##_registrar(#name, name); \
    static void name()
#define EXPECT_TRUE(c) \
    do { if (!(c)) mini::fail(__FILE__, __LINE__, "EXPECT_TRUE(" #c ")"); } while (0)
#define EXPECT_EQ(a, b) \
    do { if (!((a) == (b))) mini::fail(__FILE__, __LINE__, "EXPECT_EQ(" #a ", " #b ")"); } while (0)
#define EXPECT_NEAR(a, b, tol) \
    do { if (!(std::fabs((a) - (b)) <= (tol))) \
        mini::fail(__FILE__, __LINE__, "EXPECT_NEAR(" #a ", " #b ", " #tol ")"); } while (0)

// ---------------- code under test -------------------------------------
// Welford's online mean/variance plus min/max, one pass, O(1) memory.
struct RunningStats {
    long long n = 0;
    double mean = 0.0, m2 = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();

    void push(double x) {
        ++n;
        const double delta = x - mean;
        mean += delta / static_cast<double>(n);
        m2 += delta * (x - mean);
        if (x < min) min = x;
        if (x > max) max = x;
    }
    // Sample variance (n-1); NaN when fewer than two values.
    double variance() const {
        return n < 2 ? std::numeric_limits<double>::quiet_NaN()
                     : m2 / static_cast<double>(n - 1);
    }
};

// ---------------- tests ----------------------------------------------------
TEST(running_stats_basic) {
    RunningStats s;
    for (double v : {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}) s.push(v);
    EXPECT_EQ(s.n, 8);
    EXPECT_NEAR(s.mean, 5.0, 1e-12);
    EXPECT_NEAR(s.variance(), 32.0 / 7.0, 1e-12);  // sum sq dev = 32, n-1 = 7
    EXPECT_EQ(s.min, 2.0);
    EXPECT_EQ(s.max, 9.0);
}

TEST(running_stats_single) {
    RunningStats s;
    s.push(3.5);
    EXPECT_NEAR(s.mean, 3.5, 1e-12);
    EXPECT_TRUE(std::isnan(s.variance()));
    EXPECT_EQ(s.min, s.max);
}

TEST(running_stats_negative) {
    RunningStats s;
    for (double v : {-1.0, -2.0, -3.0}) s.push(v);
    EXPECT_NEAR(s.mean, -2.0, 1e-12);
    EXPECT_NEAR(s.variance(), 1.0, 1e-12);
    EXPECT_EQ(s.min, -3.0);
    EXPECT_EQ(s.max, -1.0);
}

TEST(running_stats_empty_is_nan) {
    RunningStats s;
    EXPECT_EQ(s.n, 0);
    EXPECT_TRUE(std::isnan(s.variance()));
}

int main() { return mini::run_all() != 0 ? 1 : 0; }
