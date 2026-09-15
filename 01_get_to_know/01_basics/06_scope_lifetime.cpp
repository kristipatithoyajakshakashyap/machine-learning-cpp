// 01_basics/06_scope_lifetime.cpp
// Scope and lifetime: where variables live and how long they live.
//
// WHAT YOU LEARN:
//   - A variable only exists inside the { } block where it is declared.
//   - A global variable (declared outside main) is visible everywhere below.
//   - Blocks create inner scopes; inner names can shadow outer ones.
//   - A static local variable keeps its value BETWEEN calls.
//   - After a block ends, its variables are destroyed.
//
// EXPECTED OUTPUT:
//   global visible inside main: 10
//   local total (sum 1..5) = 15
//   inside the block, secret = 7
//   shadowed inner value = 2 (outer was 1)
//   outer value back to 1
//   static counter calls: 1 2 3
//   while loop ended with i = 3

#include <iostream>

int global_value = 10;  // global: visible everywhere below this line

// Static local variable: remembers its value between calls.
void next_call() {
    static int calls = 0;
    ++calls;
    std::cout << " " << calls;
}

int main() {
    std::cout << "global visible inside main: " << global_value << "\n";

    // --- local variable in main ---
    int total = 0;
    for (int number = 1; number <= 5; ++number) {
        total += number;
    }
    std::cout << "local total (sum 1..5) = " << total << "\n";

    // --- inner block: the variable only lives inside the braces ---
    {
        int secret = 7;
        std::cout << "inside the block, secret = " << secret << "\n";
    }  // 'secret' is destroyed here; using it below would be an error

    // --- shadowing: inner { } can reuse a name (avoid in real code) ---
    {
        int value = 1;
        {
            int value = 2;  // shadows the outer one inside here only
            std::cout << "shadowed inner value = " << value
                      << " (outer was 1)\n";
        }
        std::cout << "outer value back to " << value << "\n";
    }

    // --- static local: keeps its value between calls ---
    std::cout << "static counter calls:";
    next_call();
    next_call();
    next_call();
    std::cout << "\n";

    // --- loop counter: its scope is the whole while statement ---
    int i = 0;
    while (i < 3) {
        ++i;
    }
    std::cout << "while loop ended with i = " << i << "\n";

    return 0;
}