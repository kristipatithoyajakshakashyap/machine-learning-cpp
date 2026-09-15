// 09_advanced_language/04_const_correctness.cpp
// const-correctness: promise, with the compiler as watchdog.
//
// WHAT YOU LEARN:
//   - T&  means "I will change it". const T& means "I only read it".
//   - A const object can only satisfy const const&-parameters — the
//     compiler refuses the reverse at compile time.
//   - A temporary (Point{5,12}) can bind to a const& but NOT to a plain &.
//   - const member functions (fun() const) promise not to mutate *this.
//
// EXPECTED OUTPUT:
//   scale((3,4), 2.0) (6,8)
//   magnitude of const (3,4) = 5
//   magnitude of temporary (5,12) = 13
//   const range-for over pts : 1 2 3

#include <cmath>      // std::sqrt
#include <iostream>
#include <vector>

struct Point {
    double x;
    double y;

    double distance_from_origin() const {      // const member: read-only
        return magnitude(x, y);
    }

    static double magnitude(double px, double py) {
        return sqrt(px * px + py * py);
    }
};

// Read-only: works with plain, const, AND temporary Points.
void show(const Point& p, const char* label) {
    std::cout << label << " (" << p.x << "," << p.y << ")\n";
}

// Read-only computation, const-reference input.
double magnitude(const Point& p) {
    return p.distance_from_origin();
}

// Mutates through a NON-const reference.
void scale(Point& p, double factor) {
    p.x *= factor;
    p.y *= factor;
}

int main() {
    Point p{3, 4};
    scale(p, 2.0);                     // needs a mutable object
    show(p, "scale((3,4), 2.0)");

    const Point q{3, 4};               // const object
    std::cout << "magnitude of const (3,4) = " << magnitude(q) << "\n";

    std::cout << "magnitude of temporary (5,12) = "
              << magnitude(Point{5, 12}) << "\n";   // temp binds const&

    std::vector<int> pts{1, 2, 3};
    std::cout << "const range-for over pts : ";
    for (const auto& x : pts) {        // read loop, no copies, no writes
        std::cout << x << " ";
    }
    std::cout << "\n";
    return 0;
}