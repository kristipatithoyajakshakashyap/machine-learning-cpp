// 02_memory_management/03_references.cpp
// References: a second name (alias) for a variable that already exists.
//
// WHAT YOU LEARN:
//   - int& r = a;  makes r an alias of a: they are THE SAME variable.
//   - Changing r changes a, and vice versa.
//   - A reference must be given someone to be "attached" to at birth.
//   - Passing a function parameter by reference lets it change the caller's
//     value and avoids copying big objects.
//   - A const reference is a read-only view with no copying.
//
// EXPECTED OUTPUT:
//   a is now 20 (set through its alias)
//   before swap_ints: 3 7 ; after swap_ints: 7 3
//   length("C++ for beginners") = 17 (read-only, no copy made)
//   nickname and name agree: C++ for beginners

#include <iostream>
#include <string>

void swap_ints(int& x, int& y) {
    int temp = x;   // references change the caller's variables directly
    x = y;
    y = temp;
}

int length(const std::string& s) {   // const ref: read but cannot change
    return static_cast<int>(s.size());
}

int main() {
    // --- a reference is an alias ---
    int a = 10;
    int& alias = a;          // alias is a second name for a
    alias = 20;              // a becomes 20 too
    std::cout << "a is now " << a << " (set through its alias)\n";

    // --- pass by reference to modify the caller's values ---
    int p = 3;
    int q = 7;
    swap_ints(p, q);
    std::cout << "before swap_ints: 3 7 ; after swap_ints: " << p << " "
              << q << "\n";

    // --- const reference: efficient read-only access ---
    std::string name = "C++ for beginners";
    std::cout << "length(\"C++ for beginners\") = " << length(name)
              << " (read-only, no copy made)\n";

    // --- both names agree ---
    std::string& nickname = name;
    std::cout << "nickname and name agree: " << nickname << "\n";

    return 0;
}