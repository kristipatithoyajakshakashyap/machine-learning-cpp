// 02_memory_management/04_dynamic_memory.cpp
// Dynamic memory: asking for memory while the program is running.
//
// WHAT YOU LEARN:
//   - new int(...) creates a value on the HEAP and gives back its address.
//   - delete frees a single heap value; delete[] frees a heap array.
//   - New and delete MUST be paired, otherwise you leak memory.
//   - After delete, set the pointer to nullptr so you do not use it twice.
//   - Heap memory is for things whose size you only know at runtime.
//
// EXPECTED OUTPUT:
//   heap int holds 7
//   heap array values: 10 20 30 40 50
//   after delete, pointers were set back to nullptr (no double delete)
//
// (The "stack vs heap" notes live in the comment block at the bottom,
//  they are not printed.)

#include <iostream>

int main() {
    // --- a single heap value ---
    int* p = new int(7);         // one int on the heap, value 7
    std::cout << "heap int holds " << *p << "\n";
    delete p;                    // give the memory back
    p = nullptr;                 // forget the old address

    // --- a heap array ---
    int* arr = new int[5];       // 5 ints on the heap (size fixed at runtime here)
    for (int i = 0; i < 5; ++i) {
        arr[i] = (i + 1) * 10;   // 10 20 30 40 50
    }
    std::cout << "heap array values:";
    for (int i = 0; i < 5; ++i) {
        std::cout << " " << arr[i];
    }
    std::cout << "\n";
    delete[] arr;                // arrays need delete[] (with the brackets)
    arr = nullptr;

    std::cout << "after delete, pointers were set back to nullptr "
                 "(no double delete)\n";

    return 0;
}

// MENTAL MODEL
//   stack (notepad):  local variables live on the stack. Fast, small,
//                     automatically cleaned when a { } block ends.
//                     Size must be known when writing the code.
//   heap  (workbench): new/delete provide memory that OUTLIVES the
//                     current block and whose size can change at runtime.
//                     Slow to grow, and YOU must remember to delete.
//
// RULES OF THUMB
//   1. Every new must pair with a delete; every new[] with a delete[].
//   2. After deleting, set the pointer back to nullptr.
//   3. Never delete a pointer you did not get from new.
//   4. Prefer smart pointers (next lesson) over raw new/delete.