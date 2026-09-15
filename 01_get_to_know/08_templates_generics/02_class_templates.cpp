// 08_templates_generics/02_class_templates.cpp
// Class templates: one class that can be instantiated for many types.
//
// WHAT YOU LEARN:
//   - template <typename T> class ... defines a family of classes.
//   - Box<int>, Box<double>, Box<string> are THREE different types, each
//     made from the same blueprint at compile time.
//   - The type is fixed at construction ("Box<int> bi(99);").
//   - Member functions are defined inside the class here; the get()/set()
//     pattern is the safe way to read and change the stored value.
//
// EXPECTED OUTPUT:
//   Box<int>   holds 99
//   Box<double> holds 2.5
//   Box<string> holds hi
//   after bi.set(50) the box holds 50
//   pair box: (42, answer)

#include <iostream>
#include <string>

// A tiny "value box": stores one T, lets you read it and replace it.
template <typename T>
class Box {
public:
    explicit Box(const T& value) : value_(value) {}

    T get() const { return value_; }
    void set(const T& new_value) { value_ = new_value; }

private:
    T value_;
};

// A pair of two possibly different types (like std::pair, simplified).
template <typename First, typename Second>
class PairBox {
public:
    PairBox(const First& a, const Second& b) : a_(a), b_(b) {}

    First  first() const { return a_; }
    Second second() const { return b_; }

private:
    First  a_;
    Second b_;
};

int main() {
    Box<int> bi(99);
    std::cout << "Box<int>   holds " << bi.get() << "\n";

    Box<double> bd(2.5);
    std::cout << "Box<double> holds " << bd.get() << "\n";

    Box<std::string> bs("hi");
    std::cout << "Box<string> holds " << bs.get() << "\n";

    bi.set(50);                              // replace the stored value
    std::cout << "after bi.set(50) the box holds " << bi.get() << "\n";

    PairBox<int, std::string> pb(42, "answer");
    std::cout << "pair box: (" << pb.first() << ", " << pb.second()
              << ")\n";
    return 0;
}