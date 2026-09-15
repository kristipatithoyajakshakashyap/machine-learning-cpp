// 08_templates_generics/01_function_templates.cpp
// Function templates: ONE function definition, MANY types.
//
// WHAT YOU LEARN:
//   - template <typename T> says "write this for whatever T the caller
//     passes".
//   - max_of must be declared BEFORE it is used (like a normal function).
//   - The compiler creates a separate copy for int, double, string, ...
//   - T const& avoids copying big objects; the < comparison drives it.
//   - The same template also works for std::vector<T>.

// EXPECTED OUTPUT:
//   max_of(3, 7)             : 7
//   max_of(2.5, 8.1)         : 8.1
//   max_of("pear","peach")   : pear
//   show(vector<int>)        : 1 2 3
//   show(vector<string>)     : a b c

#include <iostream>
#include <string>
#include <vector>

// Return the larger of two values, whatever type they have.
template <typename T>
T max_of(const T& a, const T& b) {
    return a < b ? b : a;
}

// Print every element of any vector<T>.
template <typename T>
void show(const std::vector<T>& v) {
    for (const T& x : v) {
        std::cout << x << " ";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "max_of(3, 7)             : " << max_of(3, 7) << "\n";
    std::cout << "max_of(2.5, 8.1)         : " << max_of(2.5, 8.1)
              << "\n";
    // string literals are const char[6]/[7]; wrap them so both are
    // std::string and the template can match them to one T.
    std::cout << "max_of(\"pear\",\"peach\")   : "
              << max_of(std::string("pear"), std::string("peach")) << "\n";

    std::vector<int> nums{1, 2, 3};
    std::cout << "show(vector<int>)        : ";
    show(nums);

    std::vector<std::string> words{"a", "b", "c"};
    std::cout << "show(vector<string>)     : ";
    show(words);

    return 0;
}