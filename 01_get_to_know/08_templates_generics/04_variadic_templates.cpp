// 08_templates_generics/04_variadic_templates.cpp
// Variadic templates: ONE function that accepts any number of arguments.
//
// WHAT YOU LEARN:
//   - ... (the ellipsis) captures "the rest of the arguments".
//   - A single-argument overload is the base case that ends recursion.
//   - In C++17 a FOLD EXPRESSION `(xs + ... + 0)` sums a whole pack in
//     one line — no recursion needed.
//   - print_args peels off one argument per call and recurses, so a pack
//     of mixed types prints cleanly.
//
// EXPECTED OUTPUT:
//   recursive_sum(1, 2, 3, 4)        = 10
//   recursive_sum(1.5, 2.5, 1)       = 4
//   fold_sum(2, 3, 4)                = 9
//   fold_sum(1.5, 2.5, 1.0)          = 5
//   fold_join("a","b","c","d")       = "abcd"
//   print_args(1, 2.5, "hi")         : 1 2.5 hi

#include <iostream>
#include <string>

// --- recursion based (the classic pre-C++17 way) ---
int recursive_sum(int only) {                    // base case: one arg
    return only;
}

template <typename... Rest>
int recursive_sum(int first, Rest... rest) {
    return first + recursive_sum(rest...);       // peel one arg, recurse
}

// --- fold expression (C++17, one line) ---
template <typename... Ts>
auto fold_sum(Ts... values) {
    return (values + ... + 0);                   // add all, start at 0
}

template <typename... Ts>
std::string fold_join(Ts... parts) {
    return (parts + ...);                        // concatenate all strings
}

// --- print any number of mixed-type arguments ---
void print_args() {}                             // base case: nothing left

template <typename First, typename... Rest>
void print_args(First first, Rest... rest) {
    std::cout << first << " ";
    print_args(rest...);
}

int main() {
    std::cout << "recursive_sum(1, 2, 3, 4)        = "
              << recursive_sum(1, 2, 3, 4) << "\n";
    std::cout << "recursive_sum(1.5, 2.5, 1)       = "
              << recursive_sum(1.5, 2.5, 1) << "\n";   // ints chop the .5

    std::cout << "fold_sum(2, 3, 4)                = "
              << fold_sum(2, 3, 4) << "\n";
    std::cout << "fold_sum(1.5, 2.5, 1.0)          = "
              << fold_sum(1.5, 2.5, 1.0) << "\n";

    std::cout << "fold_join(\"a\",\"b\",\"c\",\"d\")       = \""
              << fold_join(std::string("a"), std::string("b"),
                           std::string("c"), std::string("d"))
              << "\"\n";

    std::cout << "print_args(1, 2.5, \"hi\")         : ";
    print_args(1, 2.5, "hi");
    std::cout << "\n";
    return 0;
}