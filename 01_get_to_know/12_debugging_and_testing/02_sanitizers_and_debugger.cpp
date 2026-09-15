// 12_debugging_and_testing/02_sanitizers_and_debugger.cpp
// A bounded buffer with contract checks, plus a recipe for sanitizers + gdb.
//
// WHAT YOU LEARN:
//   - Contract checks (throw on out-of-range / empty / full) turn silent
//     memory corruption into a loud, catchable error.
//   - AddressSanitizer (-fsanitize=address) catches out-of-bounds and
//     use-after-free; UBSan (-fsanitize=undefined) catches overflow, bad
//     shifts, null dereference. Both need -g for readable reports.
//   - gdb basics: break <function>, run, bt, print <expr>, next, step.
//
// EXPECTED OUTPUT:
//   push 10 20 30 -> size 3 / capacity 4
//   at(1) = 20
//   pop -> 10, size now 2
//   contract violation caught: BoundedBuffer::at index 5 >= size 2
//   contract violation caught: BoundedBuffer::push buffer is full
//   contract violation caught: BoundedBuffer::pop buffer is empty
//   --- how to reproduce with sanitizers ---
//   (recipe lines)
//   all checks passed

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Fixed-capacity FIFO buffer. Every public operation validates its inputs.
class BoundedBuffer {
public:
    explicit BoundedBuffer(size_t capacity) : capacity_(capacity) {
        if (capacity == 0) throw std::invalid_argument("BoundedBuffer capacity must be > 0");
    }

    void push(int value) {
        if (data_.size() >= capacity_) throw std::out_of_range("BoundedBuffer::push buffer is full");
        data_.push_back(value);
    }

    int pop() {
        if (data_.empty()) throw std::out_of_range("BoundedBuffer::pop buffer is empty");
        const int front = data_.front();
        data_.erase(data_.begin());
        return front;
    }

    int at(size_t index) const {
        if (index >= data_.size()) {
            throw std::out_of_range("BoundedBuffer::at index " + std::to_string(index) +
                                    " >= size " + std::to_string(data_.size()));
        }
        return data_[index];
    }

    size_t size() const { return data_.size(); }
    size_t capacity() const { return capacity_; }

private:
    size_t capacity_;
    std::vector<int> data_;
};

// Runs `action`, expects it to throw std::out_of_range, prints the message.
template <typename F>
bool expect_violation(F action) {
    try {
        action();
    } catch (const std::out_of_range& e) {
        std::cout << "contract violation caught: " << e.what() << '\n';
        return true;
    }
    std::cout << "ERROR: expected a contract violation\n";
    return false;
}

void print_recipe() {
    std::cout << "--- how to reproduce with sanitizers ---\n"
              << "  # build a diagnostic binary (GCC/Clang, Linux/macOS/WSL):\n"
              << "  g++ -std=c++17 -g -O1 -fno-omit-frame-pointer \\\n"
              << "      -fsanitize=address,undefined 02_sanitizers_and_debugger.cpp -o san\n"
              << "  ./san            # ASan/UBSan print a stack trace on the first bug\n"
              << "  # step through it in gdb:\n"
              << "  gdb ./san\n"
              << "  (gdb) break BoundedBuffer::at\n"
              << "  (gdb) run\n"
              << "  (gdb) bt                 # who called at()?\n"
              << "  (gdb) print index\n"
              << "  (gdb) print data_.size()\n"
              << "  (gdb) next               # step over; 'step' goes into calls\n"
              << "  (gdb) continue\n"
              << "  # note: MinGW g++ on Windows lacks libasan; use WSL or clang-cl.\n";
}

int main() {
    int failures = 0;
    BoundedBuffer buf(4);
    buf.push(10);
    buf.push(20);
    buf.push(30);
    std::cout << "push 10 20 30 -> size " << buf.size() << " / capacity "
              << buf.capacity() << '\n';
    std::cout << "at(1) = " << buf.at(1) << '\n';
    if (buf.at(1) != 20) ++failures;
    const int popped = buf.pop();
    std::cout << "pop -> " << popped << ", size now " << buf.size() << '\n';
    if (popped != 10 || buf.size() != 2) ++failures;

    if (!expect_violation([&] { buf.at(5); })) ++failures;
    if (!expect_violation([&] {
            buf.push(1);
            buf.push(2);
            buf.push(3);  // fifth element into capacity 4
        })) ++failures;
    if (!expect_violation([&] {
            BoundedBuffer empty(1);
            empty.pop();
        })) ++failures;

    print_recipe();
    std::cout << (failures == 0 ? "all checks passed\n" : "SOME CHECKS FAILED\n");
    return failures == 0 ? 0 : 1;
}
