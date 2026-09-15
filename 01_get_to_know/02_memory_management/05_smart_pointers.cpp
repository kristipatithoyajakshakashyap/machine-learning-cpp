// 02_memory_management/05_smart_pointers.cpp
// Smart pointers: let the program delete memory for you (no new/delete).
//
// WHAT YOU LEARN:
//   - unique_ptr: one owner. Nobody else may own it. Freed when it dies.
//   - shared_ptr: many owners. Freed when the LAST owner dies.
//   - weak_ptr:   a visitor that does not keep anything alive.
//   - make_unique / make_shared are the safe way to create them.
//
// EXPECTED OUTPUT:
//   a unique_ptr owns the value 5
//   shared value is 10 while both owners are alive (use_count 2)
//   one owner went out of scope -> use_count now 1
//   weak_ptr.lock() gives a shared_ptr while the owner is alive
//   heavy object: a shared object with no leaks, no manual delete

#include <iostream>
#include <memory>
#include <string>

// A tiny object that says when it is created and destroyed.
struct Toy {
    explicit Toy(std::string n) : name(n) {
        std::cout << "  Toy created: " << name << "\n";
    }
    ~Toy() {
        std::cout << "  Toy destroyed: " << name << "\n";
    }
    std::string name;
};

int main() {
    // --- unique_ptr: one owner ---
    auto u = std::make_unique<int>(5);
    std::cout << "a unique_ptr owns the value " << *u << "\n";
    // no delete needed: u frees its int automatically when main ends

    // --- shared_ptr: many owners; freed when the last one leaves ---
    std::shared_ptr<int> s = std::make_shared<int>(10);
    {
        std::shared_ptr<int> copy = s;     // second owner
        std::cout << "shared value is " << *s
                  << " while both owners are alive (use_count "
                  << s.use_count() << ")\n";
    }                                      // 'copy' destroyed here
    std::cout << "one owner went out of scope -> use_count now "
              << s.use_count() << "\n";

    // --- weak_ptr: watches without keeping alive ---
    std::weak_ptr<int> watcher = s;
    if (std::shared_ptr<int> locked = watcher.lock()) {
        std::cout << "weak_ptr.lock() gives a shared_ptr while the owner is alive ("
                  << *locked << ")\n";
    }
    s.reset();                             // drop the last owner
    if (watcher.lock() == nullptr) {
        std::cout << "after the owner dies, weak_ptr.lock() returns empty\n";
    }

    // --- plain shared object lifecycle (no manual delete anywhere) ---
    std::cout << "heavy object:\n";
    auto toy = std::make_shared<Toy>("rubber duck");
    std::cout << "  toy in use\n";

    return 0;
}