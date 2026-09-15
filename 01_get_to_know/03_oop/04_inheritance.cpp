// 03_oop/04_inheritance.cpp
// Inheritance: building a new class on top of an existing one.
//
// WHAT YOU LEARN:
//   - class Dog : public Animal  ->  Dog "is an" Animal (IS-A).
//   - The derived class gets the base class's public and protected members.
//   - protected: visible to the base AND to derived classes, private to all.
//   - The derived constructor calls the base constructor in the list.
//   - override lets a derived method replace a base method's behavior.
//
// EXPECTED OUTPUT:
//   Rex says Woof!
//   Tom says Meow!
//   A Dog is an Animal (inheritance = the "is-a" relationship)

#include <iostream>
#include <string>

class Animal {
public:
    explicit Animal(const std::string& n) : name(n) {}

    void describe() const {
        std::cout << name << " says " << speak() << "\n";
    }

    // virtual: the "real" version is chosen at runtime
    virtual std::string speak() const {
        return "...";
    }

protected:
    std::string name;   // visible to Animal AND to Dog/Cat
};

class Dog : public Animal {
public:
    explicit Dog(const std::string& n) : Animal(n) {}

    std::string speak() const override {
        return "Woof!";
    }
};

class Cat : public Animal {
public:
    explicit Cat(const std::string& n) : Animal(n) {}

    std::string speak() const override {
        return "Meow!";
    }
};

int main() {
    Dog rex("Rex");
    Cat tom("Tom");

    rex.describe();       // prints "Rex says Woof!"
    tom.describe();       // prints "Tom says Meow!"

    std::cout << "A Dog is an Animal (inheritance = the \"is-a\" "
                 "relationship)\n";
    return 0;
}