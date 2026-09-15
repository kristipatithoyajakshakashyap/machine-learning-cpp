// 08_templates_generics/05_type_traits_sfinae.cpp
// Asking the compiler questions about types: <type_traits>, SFINAE, the
// detection idiom, and how if constexpr replaces most of it in C++17.
//
// WHAT YOU LEARN:
//   - Type traits are compile-time predicates: std::is_same_v<A,B>,
//     std::is_integral_v<T>, std::is_floating_point_v<T>.
//   - std::decay_t strips references/const/arrays; std::conditional_t picks
//     one of two types from a boolean.
//   - SFINAE ("substitution failure is not an error"): std::enable_if_t in
//     a return type removes an overload from consideration instead of
//     causing a compile error, so integral and floating overloads coexist.
//   - The detection idiom (std::void_t) builds a trait such as has_size<T>
//     that is true exactly when `t.size()` is a valid expression.
//   - if constexpr does the same job inside ONE function body and is usually
//     the clearer choice in C++17. static_assert documents the answers.
//
// EXPECTED OUTPUT:
//   is_same<int, int>           = true
//   is_same<int, long>          = false
//   is_integral<char>           = true
//   is_floating_point<float>    = true
//   decay<const int&>  is int   = true
//   decay<int[4]>      is int*  = true
//   conditional<(4 > 2)>        = true
//   halve(7)   [integral overload] -> 3
//   halve(7.0) [floating overload] -> 3.5
//   has_size<vector<int>>  = true
//   has_size<std::string>  = true
//   has_size<int>          = false
//   length(vector of 3)    = 3
//   length("hello")        = 5
//   length(42)             = 1
//   describe(3)     -> integral
//   describe(3.5)   -> floating
//   describe("str") -> other

#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

// --- SFINAE with enable_if: two overloads, exactly one survives -----------
template <typename T>
std::enable_if_t<std::is_integral_v<T>, T> halve(T x) {
    return x / 2;                       // integer division
}

template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, T> halve(T x) {
    return x / T{2};                    // floating division
}

// --- detection idiom: does T have a .size() member? -----------------------
template <typename T, typename = void>
struct has_size : std::false_type {};

template <typename T>
struct has_size<T, std::void_t<decltype(std::declval<const T&>().size())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_size_v = has_size<T>::value;

// Uses the trait through if constexpr: one function, three behaviours.
template <typename T>
std::size_t length(const T& value) {
    if constexpr (has_size_v<T>) {
        return value.size();
    } else if constexpr (std::is_array_v<T>) {
        return std::extent_v<T> - 1;    // string literal: drop the '\0'
    } else {
        return 1;                       // a scalar "has one element"
    }
}

// --- if constexpr as the C++17 replacement for enable_if -------------------
template <typename T>
std::string describe(const T&) {
    using U = std::decay_t<T>;          // treat "str" (const char[4]) sanely
    if constexpr (std::is_integral_v<U>) {
        return "integral";
    } else if constexpr (std::is_floating_point_v<U>) {
        return "floating";
    } else {
        return "other";
    }
}

// --- static_assert: the traits are checked before main even runs ----------
static_assert(std::is_same_v<std::decay_t<const int&>, int>);
static_assert(std::is_same_v<std::conditional_t<true, int, double>, int>);
static_assert(has_size_v<std::vector<int>> && !has_size_v<int>);

const char* tf(bool b) { return b ? "true" : "false"; }

int main() {
    std::cout << std::boolalpha;
    std::cout << "is_same<int, int>           = " << tf(std::is_same_v<int, int>) << '\n';
    std::cout << "is_same<int, long>          = " << tf(std::is_same_v<int, long>) << '\n';
    std::cout << "is_integral<char>           = " << tf(std::is_integral_v<char>) << '\n';
    std::cout << "is_floating_point<float>    = " << tf(std::is_floating_point_v<float>) << '\n';
    std::cout << "decay<const int&>  is int   = "
              << tf(std::is_same_v<std::decay_t<const int&>, int>) << '\n';
    std::cout << "decay<int[4]>      is int*  = "
              << tf(std::is_same_v<std::decay_t<int[4]>, int*>) << '\n';
    using Picked = std::conditional_t<(4 > 2), int, double>;
    std::cout << "conditional<(4 > 2)>        = " << tf(std::is_same_v<Picked, int>) << '\n';

    std::cout << "halve(7)   [integral overload] -> " << halve(7) << '\n';
    std::cout << "halve(7.0) [floating overload] -> " << halve(7.0) << '\n';

    std::cout << "has_size<vector<int>>  = " << tf(has_size_v<std::vector<int>>) << '\n';
    std::cout << "has_size<std::string>  = " << tf(has_size_v<std::string>) << '\n';
    std::cout << "has_size<int>          = " << tf(has_size_v<int>) << '\n';

    const std::vector<int> v{1, 2, 3};
    std::cout << "length(vector of 3)    = " << length(v) << '\n';
    std::cout << "length(\"hello\")        = " << length("hello") << '\n';
    std::cout << "length(42)             = " << length(42) << '\n';

    std::cout << "describe(3)     -> " << describe(3) << '\n';
    std::cout << "describe(3.5)   -> " << describe(3.5) << '\n';
    std::cout << "describe(\"str\") -> " << describe("str") << '\n';
    return 0;
}
