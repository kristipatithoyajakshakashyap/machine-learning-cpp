// 02_memory_management/02_pointers.cpp
// Pointers: variables that store the memory ADDRESS of another thing.
//
// WHAT YOU LEARN:
//   - &x  is "the address of x".
//   - int* p stores an address; *p reads or writes whatever p points at.
//   - A null pointer (nullptr) points nowhere; always test it first.
//   - Pointer arithmetic moves between elements of an array.
//   - A function can change a variable in main by receiving its address.
//
// EXPECTED OUTPUT:
//   x is 42, and the pointer reads 42 through it
//   writing *p = 99 changed x to 99
//   a null pointer can be tested safely (nothing == nullptr)
//   array values: 10 20 30 40
//   *(p + 2) is 30 (third element)
//   after add_ten(&x), x is 109

#include <iostream>

// A function that changes the caller's variable using a pointer.
void add_ten(int* v) {
    *v += 10;   // write through the pointer: modifies the original
}

int main() {
    // --- address-of (&) and pointer (*) basics ---
    int x = 42;
    int* p = &x;                 // p holds the address of x
    std::cout << "x is " << x << ", and the pointer reads "
              << *p << " through it\n";

    *p = 99;                     // change x by writing through p
    std::cout << "writing *p = 99 changed x to " << x << "\n";

    // --- null pointers ---
    int* nothing = nullptr;      // points nowhere
    if (nothing == nullptr) {
        std::cout << "a null pointer can be tested safely "
                     "(nothing == nullptr)\n";
    }

    // --- pointer arithmetic over an array ---
    int arr[4] = {10, 20, 30, 40};
    int* p0 = &arr[0];
    std::cout << "array values:";
    for (int i = 0; i < 4; ++i) {
        std::cout << " " << *(p0 + i);   // move pointer i steps forward
    }
    std::cout << "\n";
    std::cout << "*(p + 2) is " << *(p0 + 2) << " (third element)\n";

    // --- function modifying main's variable by address ---
    add_ten(&x);
    std::cout << "after add_ten(&x), x is " << x << "\n";

    return 0;
}