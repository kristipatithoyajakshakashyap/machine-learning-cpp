  // 03_oop/06_operator_overloading.cpp
// Operator overloading: teaching +, ==, and << to your own classes.
//
// WHAT YOU LEARN:
//   - +  == << are just functions with a special name.
//   - Adding operator+ lets you write a + b for YOUR type.
//   - += is implemented in the class (it changes *this).
//   - operator<< lets std::cout print your object.
//   - friend functions may read the private members of a class.
//
// EXPECTED OUTPUT:
//   a = (1, 2)
//   b = (3, 4)
//   a + b = (4, 6)
//   after a += b, a = (4, 6)
//   a equals a freshly built (4, 6)? true

#include <iostream>

class Vector2D {
public:
    Vector2D(double a, double b) : x(a), y(b) {}

    // += changes THIS object and returns a reference to it
    Vector2D& operator+=(const Vector2D& o) {
        x += o.x;
        y += o.y;
        return *this;
    }

    // friend functions can read the private x and y below
    friend std::ostream& operator<<(std::ostream& os, const Vector2D& v);
    friend bool operator==(const Vector2D& a, const Vector2D& b);

private:
    double x;
    double y;
};

// + as a free function: copy, add, return the copy
Vector2D operator+(Vector2D a, const Vector2D& b) {
    a += b;    // reuse += instead of duplicating the math
    return a;
}

// print (x, y)
std::ostream& operator<<(std::ostream& os, const Vector2D& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}

// compare two vectors
bool operator==(const Vector2D& a, const Vector2D& b) {
    return a.x == b.x && a.y == b.y;
}

int main() {
    Vector2D a(1, 2);
    Vector2D b(3, 4);

    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n";

    Vector2D c = a + b;
    std::cout << "a + b = " << c << "\n";

    a += b;
    std::cout << "after a += b, a = " << a << "\n";

    Vector2D d(4, 6);
    std::cout << "a equals a freshly built (4, 6)? " << std::boolalpha
              << (a == d) << "\n";

    return 0;
}