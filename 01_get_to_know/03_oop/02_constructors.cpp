// 03_oop/02_constructors.cpp
// Constructors: the special functions that build an object.
//
// WHAT YOU LEARN:
//   - A constructor runs AUTOMATICALLY when an object is created.
//   - The initializer list (name(n), id(i)) fills members at birth.
//   - You can have several constructors (default, with arguments, copy).
//   - A copy constructor makes a new object that is a copy of another.
//
// EXPECTED OUTPUT:
//   made with default: unknown (0)
//   made with arguments: Ada (101)
//   a copy of Ada: Ada (101)
//   copies are independent:
//   changing the copy leaves the original untouched -> Ada (101)

#include <iostream>
#include <string>

class Student {
public:
    // 1) default constructor: called when no arguments are given
    Student() : name("unknown"), id(0) {}

    // 2) constructor with arguments
    Student(const std::string& n, int i) : name(n), id(i) {}

    // 3) copy constructor: build a copy of another object
    Student(const Student& other) : name(other.name), id(other.id) {}

    void print() const {
        std::cout << name << " (" << id << ")";
    }

    void rename(const std::string& n) { name = n; }

private:
    std::string name;
    int id;
};

int main() {
    Student a;                  // default constructor
    std::cout << "made with default: ";
    a.print();
    std::cout << "\n";

    Student b("Ada", 101);      // constructor with arguments
    std::cout << "made with arguments: ";
    b.print();
    std::cout << "\n";

    Student c = b;              // copy constructor
    std::cout << "a copy of Ada: ";
    c.print();
    std::cout << "\n";

    // copies are independent: changing c does not change b
    c.rename("Grace");
    std::cout << "changing the copy leaves the original untouched -> ";
    b.print();
    std::cout << "\n";

    return 0;
}