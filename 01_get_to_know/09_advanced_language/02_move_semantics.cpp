// 09_advanced_language/02_move_semantics.cpp
// Move semantics: steal resources instead of copying them.
//
// WHAT YOU LEARN:
//   - std::move(x) turns x into an rvalue: "you may take my internals".
//   - Moving a std::string transfers its heap buffer — no deep copy.
//   - Move ASSIGNMENT (v1 = std::move(v2)) swaps owners instantly.
//   - A moved-from object is left VALID but UNSPECIFIED: you may assign
//     to it or let it die — you may NOT assume what it holds.
//   - This is why the Rule of Five (03_oop) asks classes that own memory
//     to implement move constructor and move assignment.
//
// EXPECTED OUTPUT:
//   std::move on a string : b now holds 'alpha', a was left empty
//   vector move-assign    : v1 now holds '100 200 ', v2 is empty
//   rule of five classes  : hand off their resources instead of copying

#include <iostream>
#include <string>
#include <vector>

int main() {
    // --- move a string: the buffer is handed to b, not duplicated ---
    std::string a = "alpha";
    std::string b = std::move(a);           // steal a's buffer
    std::cout << "std::move on a string : b now holds '" << b
              << "', a was left empty\n";

    // --- move-assign a vector ---
    std::vector<int> v1{1, 2, 3};
    std::vector<int> v2{100, 200};
    v1 = std::move(v2);                     // take ownership of v2's array
    std::cout << "vector move-assign    : v1 now holds '";
    for (int x : v1) {
        std::cout << x << " ";
    }
    std::cout << "', v2 is empty\n";

    // --- a class that owns memory: move transfers, copy duplicates ---
    struct Owner {
        std::string data;
    };
    Owner x{"shared text"};
    Owner shallow_copy = x;                 // COPY: duplicates the string
    Owner moved = std::move(x);             // MOVE: steals x's string
    std::cout << "rule of five classes  : hand off their resources "
                 "instead of copying\n";
    std::cout << "                      moved.data = '" << moved.data
              << "'\n";
    return 0;
}