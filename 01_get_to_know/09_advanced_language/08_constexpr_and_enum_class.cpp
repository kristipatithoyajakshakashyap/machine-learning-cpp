// 09_advanced_language/08_constexpr_and_enum_class.cpp
// Compile-time values with constexpr and type-safe enumerations.
//
// WHAT YOU LEARN:
//   - constexpr functions can run at compile time; static_assert proves it.
//   - if constexpr picks a branch inside a template at compile time, so the
//     discarded branch never has to compile for that type.
//   - enum class values do not convert to int silently; give them an
//     explicit underlying type and switch over ALL cases.
//   - A bit-flag enum class needs overloaded | and & operators to combine
//     and test flags without casting everywhere.
//
// EXPECTED OUTPUT:
//   factorial(5) at compile time = 120
//   kMaxItems = 64, kHalf = 32
//   describe(42)      -> integer 42
//   describe(2.5)     -> floating 2.5
//   describe("text")  -> other
//   Color::Red   -> stop
//   Color::Green -> go
//   Color::Blue  -> ??? (not a traffic light)
//   sizeof(Color) = 1 byte(s)
//   perm = Read|Write, has Read: yes, has Exec: no
//   after adding Exec: has Exec: yes

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

// --- constexpr ------------------------------------------------------------
constexpr int factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }

constexpr int kMaxItems = 64;
constexpr int kHalf = kMaxItems / 2;

static_assert(factorial(5) == 120, "factorial(5) must be 120");
static_assert(kHalf == 32, "half of 64 is 32");

// --- if constexpr in a template -------------------------------------------
template <typename T>
std::string describe(const T& value) {
    if constexpr (std::is_integral_v<T>) {
        return "integer " + std::to_string(value);
    } else if constexpr (std::is_floating_point_v<T>) {
        return "floating " + std::to_string(value).substr(0, 3);
    } else {
        return "other";  // to_string is never instantiated for this T
    }
}

// --- enum class with underlying type --------------------------------------
enum class Color : std::uint8_t { Red, Green, Blue };

std::string traffic_meaning(Color c) {
    switch (c) {           // every enumerator handled: no -Wswitch warning
        case Color::Red:   return "stop";
        case Color::Green: return "go";
        case Color::Blue:  return "??? (not a traffic light)";
    }
    return "invalid";      // reached only for an out-of-range value
}

// --- bit-flag enum class --------------------------------------------------
enum class Perm : unsigned { None = 0, Read = 1, Write = 2, Exec = 4 };

constexpr Perm operator|(Perm a, Perm b) {
    return static_cast<Perm>(static_cast<unsigned>(a) | static_cast<unsigned>(b));
}
constexpr Perm operator&(Perm a, Perm b) {
    return static_cast<Perm>(static_cast<unsigned>(a) & static_cast<unsigned>(b));
}
constexpr bool has(Perm set, Perm flag) { return (set & flag) == flag; }

const char* yes_no(bool b) { return b ? "yes" : "no"; }

int main() {
    std::cout << "factorial(5) at compile time = " << factorial(5) << '\n';
    std::cout << "kMaxItems = " << kMaxItems << ", kHalf = " << kHalf << '\n';

    std::cout << "describe(42)      -> " << describe(42) << '\n';
    std::cout << "describe(2.5)     -> " << describe(2.5) << '\n';
    std::cout << "describe(\"text\")  -> " << describe(std::string("text")) << '\n';

    std::cout << "Color::Red   -> " << traffic_meaning(Color::Red) << '\n';
    std::cout << "Color::Green -> " << traffic_meaning(Color::Green) << '\n';
    std::cout << "Color::Blue  -> " << traffic_meaning(Color::Blue) << '\n';
    std::cout << "sizeof(Color) = " << sizeof(Color) << " byte(s)\n";

    Perm perm = Perm::Read | Perm::Write;
    std::cout << "perm = Read|Write, has Read: " << yes_no(has(perm, Perm::Read))
              << ", has Exec: " << yes_no(has(perm, Perm::Exec)) << '\n';
    perm = perm | Perm::Exec;
    std::cout << "after adding Exec: has Exec: " << yes_no(has(perm, Perm::Exec))
              << '\n';
    return 0;
}
