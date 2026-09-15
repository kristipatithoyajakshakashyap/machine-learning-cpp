// 03_oop/07_rule_of_five.cpp
// The Rule of Five: when your class owns memory, you must provide:
//   1) destructor          2) copy constructor   3) copy assignment
//   4) move constructor    5) move assignment
//
// WHAT YOU LEARN:
//   - A class that uses new/delete must manage that memory itself.
//   - COPY duplicates the memory -> both objects stay independent.
//   - MOVE steals the memory -> the source becomes empty (fast).
//   - We count copies and moves so you can SEE what happens.
//
// EXPECTED OUTPUT:
//   A = [10 20 30]
//   B copied from A        (copies so far: 1, moves so far: 0)
//   after B[0]=99, A[0] is still 10 (copies are independent)
//   C moved from A         (copies so far: 1, moves so far: 1) A is now size 0
//   D assigned from B      (copies so far: 2, moves so far: 1) D[0]=99
//   E = move(C)            (copies so far: 2, moves so far: 2) E[1]=20
//   Rule of Five in action: copies duplicate, moves steal

#include <algorithm>   // std::copy
#include <cstddef>     // std::size_t
#include <iostream>
#include <utility>     // std::move

class SmartBuffer {
public:
    // --- counters shared by EVERY SmartBuffer object ---
    static int copies_made;
    static int moves_made;

    // 0) constructor: allocate the memory
    explicit SmartBuffer(std::size_t n) : len(n), data(new int[n]()) {
    }

    // 1) destructor: give the memory back
    ~SmartBuffer() {
        delete[] data;
    }

    // 2) copy constructor: allocate fresh memory, copy every value
    SmartBuffer(const SmartBuffer& other)
        : len(other.len), data(new int[other.len]) {
        std::copy(other.data, other.data + other.len, data);
        ++copies_made;
    }

    // 3) copy assignment: free ours, then copy the other's values
    SmartBuffer& operator=(const SmartBuffer& other) {
        if (this != &other) {                  // guard self-assignment
            delete[] data;
            len = other.len;
            data = new int[len];
            std::copy(other.data, other.data + len, data);
            ++copies_made;
        }
        return *this;
    }

    // 4) move constructor: STEAL the memory, make the source empty
    SmartBuffer(SmartBuffer&& other) noexcept
        : len(other.len), data(other.data) {
        other.data = nullptr;
        other.len = 0;
        ++moves_made;
    }

    // 5) move assignment: steal it too
    SmartBuffer& operator=(SmartBuffer&& other) noexcept {
        if (this != &other) {
            delete[] data;
            len = other.len;
            data = other.data;
            other.data = nullptr;
            other.len = 0;
            ++moves_made;
        }
        return *this;
    }

    void set(std::size_t i, int v) { data[i] = v; }
    int get(std::size_t i) const  { return data[i]; }
    std::size_t size() const      { return len; }

private:
    std::size_t len;
    int* data;
};

int SmartBuffer::copies_made = 0;
int SmartBuffer::moves_made  = 0;

int main() {
    SmartBuffer a(3);
    a.set(0, 10);
    a.set(1, 20);
    a.set(2, 30);
    std::cout << "A = [" << a.get(0) << " " << a.get(1) << " " << a.get(2)
              << "]\n";

    // --- copy 1: duplicate ---
    SmartBuffer b = a;                         // copy constructor
    std::cout << "B copied from A        (copies so far: "
              << SmartBuffer::copies_made << ", moves so far: "
              << SmartBuffer::moves_made << ")\n";
    b.set(0, 99);
    std::cout << "after B[0]=99, A[0] is still " << a.get(0)
              << " (copies are independent)\n";

    // --- move 1: steal ---
    SmartBuffer c = std::move(a);              // move constructor
    std::cout << "C moved from A         (copies so far: "
              << SmartBuffer::copies_made << ", moves so far: "
              << SmartBuffer::moves_made << ") A is now size "
              << a.size() << "\n";

    // --- copy 2: via assignment ---
    SmartBuffer d(1);
    d = b;                                     // copy assignment
    std::cout << "D assigned from B      (copies so far: "
              << SmartBuffer::copies_made << ", moves so far: "
              << SmartBuffer::moves_made << ") D[0]=" << d.get(0) << "\n";

    // --- move 2: via assignment ---
    SmartBuffer e(1);
    e = std::move(c);                          // move assignment
    std::cout << "E = move(C)            (copies so far: "
              << SmartBuffer::copies_made << ", moves so far: "
              << SmartBuffer::moves_made << ") E[1]=" << e.get(1) << "\n";

    std::cout << "Rule of Five in action: copies duplicate, moves steal\n";
    return 0;
}