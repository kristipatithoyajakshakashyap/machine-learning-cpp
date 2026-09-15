// 03_oop/01_classes_objects.cpp
// Classes and objects: bundling data with the functions that use it.
//
// WHAT YOU LEARN:
//   - A class is a BLUEPRINT; an object is one real instance of it.
//   - Members = data fields; methods = functions inside the class.
//   - Access: public (anyone), private (only the class), protected (later).
//   - struct defaults to public, class defaults to private.
//
// EXPECTED OUTPUT:
//   Object 1: Ada (id 101)
//   Object 2: Grace (id 102)
//   struct members are public by default (p.x = 3.5 worked from main)
//   class members are private by default (the commented line would not compile)
//   two students exist, ids 101 and 102

#include <iostream>
#include <string>

// --- a class: data + methods ---
class Student {
public:                          // everything below is public
    std::string name;
    int id = 0;

    void print() const {         // a method (function inside the class)
        std::cout << name << " (id " << id << ")";
    }
};

// --- a struct: defaults to public ---
struct Point {
    double x = 0.0;
    double y = 0.0;
};

// --- a struct with a PRIVATE section: shows the difference ---
struct StudentSecret {
    std::string name;            // this one is public (struct default)
private:
    int id = 0;                  // this one is private
};

int main() {
    // --- creating objects and using them ---
    Student ada;
    ada.name = "Ada";
    ada.id = 101;

    Student grace;
    grace.name = "Grace";
    grace.id = 102;

    std::cout << "Object 1: ";
    ada.print();
    std::cout << "\nObject 2: ";
    grace.print();
    std::cout << "\n";

    // --- struct public members work directly ---
    Point p;
    p.x = 3.5;
    p.y = 2.0;
    std::cout << "struct members are public by default (p.x = " << p.x
              << " worked from main)\n";

    // Uncomment the two lines below to SEE the compile error:
    //   StudentSecret s;
    //   s.id = 5;                    // error: 'id' is private
    std::cout << "class members are private by default "
                 "(the commented line would not compile)\n";

    std::cout << "two students exist, ids " << ada.id << " and " << grace.id
              << "\n";

    return 0;
}