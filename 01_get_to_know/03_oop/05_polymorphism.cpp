// 03_oop/05_polymorphism.cpp
// Polymorphism: one interface, many behaviors.
//
// WHAT YOU LEARN:
//   - A base class with a "pure virtual" function (= 0) is ABSTRACT:
//     you cannot make an object of it, only of its derived classes.
//   - The same call (shape->area()) picks the right implementation
//     at runtime depending on what kind of shape you really have.
//   - A base pointer/reference can point at ANY derived object.
//
// EXPECTED OUTPUT:
//   Circle(radius 3)      area = 28.2743
//   Rectangle(4 x 5)      area = 20.0000
//   same call 'area()' gives different answers - that is polymorphism

#include <cmath>          // M_PI is a POSIX extension; use std::acos for portability
#include <iomanip>        // std::setprecision
#include <iostream>
#include <memory>
#include <vector>

// --- abstract base class (interface): defines WHAT every shape can do ---
class Shape {
public:
    virtual double area() const = 0;   // pure virtual: no body here
    virtual ~Shape() = default;        // derived dtors clean up correctly
};

class Circle : public Shape {
public:
    explicit Circle(double r) : radius(r) {}
    double area() const override {
        const double pi = std::acos(-1.0);   // this is pi, anywhere
        return pi * radius * radius;
    }
private:
    double radius;
};

class Rectangle : public Shape {
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const override {
        return width * height;
    }
private:
    double width;
    double height;
};

int main() {
    // a base-class pointer can hold an object of ANY derived class
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>(3.0));
    shapes.push_back(std::make_unique<Rectangle>(4.0, 5.0));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Circle(radius 3)      area = " << shapes[0]->area()
              << "\n"
              << "Rectangle(4 x 5)      area = " << shapes[1]->area() << "\n";

    std::cout << "same call 'area()' gives different answers - "
                 "that is polymorphism\n";
    return 0;
}