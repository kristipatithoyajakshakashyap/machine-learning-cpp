// 07_stl_core/07_array_and_string_view.cpp
// std::array (fixed size, on the stack) and std::string_view (no-copy text).
//
// WHAT YOU LEARN:
//   - std::array<T, N> is a fixed-size aggregate: size() is known at compile
//     time, it lives wherever you declare it (no heap), and it works in
//     constexpr code.
//   - Structured bindings unpack an array: auto [x, y, z] = arr;
//   - std::string_view is a (pointer, length) pair that VIEWS existing text.
//     substr() and find() never copy; passing it to functions is cheap.
//   - A string_view must not outlive the text it points at (dangling).
//
// EXPECTED OUTPUT:
//   array size = 3, elements: 1 2 3
//   structured bindings: x=1 y=2 z=3
//   constexpr sum of {1,2,3} = 6 (checked at compile time)
//   view of "hello, world": len 12
//   substr(7) = "world" (no copy)
//   find("world") = 7
//   count_vowels("hello, world") = 3
//   count_vowels via literal = 2
//   safe: view rebuilt from a live string = "still alive"
//   | type         | owns data | fixed size | heap  | resizable |
//   | std::array   | yes       | yes        | no    | no        |
//   | std::vector  | yes       | no         | yes   | yes       |
//   | std::string  | yes       | no         | maybe | yes       |
//   | string_view  | no        | n/a        | no    | no        |

#include <array>
#include <iostream>
#include <string>
#include <string_view>

// constexpr: the compiler can evaluate this at compile time.
constexpr int sum3(const std::array<int, 3>& a) { return a[0] + a[1] + a[2]; }

// Takes a view: works for std::string, string literals, and other views
// without ever copying the characters.
int count_vowels(std::string_view text) {
    int n = 0;
    for (char c : text) {
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') ++n;
    }
    return n;
}

int main() {
    // --- std::array -----------------------------------------------------
    std::array<int, 3> arr{1, 2, 3};                  // aggregate init
    std::cout << "array size = " << arr.size() << ", elements:";
    for (int v : arr) std::cout << ' ' << v;
    std::cout << '\n';

    auto [x, y, z] = arr;                              // structured bindings
    std::cout << "structured bindings: x=" << x << " y=" << y << " z=" << z
              << '\n';

    constexpr std::array<int, 3> k{1, 2, 3};
    constexpr int total = sum3(k);                     // compile-time value
    static_assert(total == 6, "sum3 must be 6");
    std::cout << "constexpr sum of {1,2,3} = " << total
              << " (checked at compile time)\n";

    // --- std::string_view -----------------------------------------------
    const std::string owner = "hello, world";          // owns the characters
    std::string_view view = owner;                     // just points at them
    std::cout << "view of \"" << view << "\": len " << view.size() << '\n';
    std::cout << "substr(7) = \"" << view.substr(7) << "\" (no copy)\n";
    std::cout << "find(\"world\") = " << view.find("world") << '\n';

    std::cout << "count_vowels(\"hello, world\") = " << count_vowels(owner)
              << '\n';
    std::cout << "count_vowels via literal = " << count_vowels("abc def")
              << '\n';

    // --- the dangling pitfall (described, not executed) -----------------
    // WRONG:  std::string_view bad = std::string("temporary");
    //         The temporary string dies at the end of that statement, so
    //         'bad' now points at freed memory -> undefined behaviour.
    // RIGHT:  keep the owner alive for as long as the view is used:
    const std::string alive = "still alive";
    std::string_view ok = alive;                       // owner outlives view
    std::cout << "safe: view rebuilt from a live string = \"" << ok << "\"\n";

    // --- comparison table -----------------------------------------------
    std::cout << "| type         | owns data | fixed size | heap  | resizable |\n"
              << "| std::array   | yes       | yes        | no    | no        |\n"
              << "| std::vector  | yes       | no         | yes   | yes       |\n"
              << "| std::string  | yes       | no         | maybe | yes       |\n"
              << "| string_view  | no        | n/a        | no    | no        |\n";
    return 0;
}
