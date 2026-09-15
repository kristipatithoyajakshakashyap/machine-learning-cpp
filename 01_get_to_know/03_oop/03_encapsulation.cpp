// 03_oop/03_encapsulation.cpp
// Encapsulation: hiding data and guarding how it changes.
//
// WHAT YOU LEARN:
//   - Keep data fields private: nobody can reach in and corrupt them.
//   - Public getters read data; public setters change it WITH validation.
//   - The setter is the only door -> bad values can be rejected or fixed.
//   - This protects "class invariants" (rules that must always hold).
//
// EXPECTED OUTPUT:
//   Ada starts at age 0
//   Ada set to 25 -> now 25
//   trying age -5 is refused, Ada stays 25
//   trying age 999 is refused, Ada stays 25
//   (impossible ages simply never happen - the setter guarded the data)

#include <iostream>
#include <string>

class Person {
public:
    explicit Person(const std::string& n) : name(n) {}

    // --- getter: read-only access ---
    int get_age() const {
        return age;
    }

    // --- setter: the ONLY way to change age, and it validates ---
    void set_age(int a) {
        if (a >= 0 && a <= 130) {   // only accept a sensible age
            age = a;
        }
    }

private:
    std::string name;
    int age = 0;                    // direct access is blocked from outside
};

int main() {
    Person ada("Ada");
    std::cout << "Ada starts at age " << ada.get_age() << "\n";

    ada.set_age(25);
    std::cout << "Ada set to 25 -> now " << ada.get_age() << "\n";

    ada.set_age(-5);                // bad value: silently refused
    std::cout << "trying age -5 is refused, Ada stays "
              << ada.get_age() << "\n";

    ada.set_age(999);               // bad value: silently refused
    std::cout << "trying age 999 is refused, Ada stays "
              << ada.get_age() << "\n";

    std::cout << "(impossible ages simply never happen - the setter "
                 "guarded the data)\n";
    return 0;
}