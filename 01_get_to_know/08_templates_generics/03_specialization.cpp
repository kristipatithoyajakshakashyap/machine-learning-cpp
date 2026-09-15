// 08_templates_generics/03_specialization.cpp
// Specialization: giving a template SPECIAL treatment for one type.
//
// WHAT YOU LEARN:
//   - The PRIMARY template is the default (here: "unknown").
//   - A FULL specialization `template <> struct TypeName<int>` replaces the
//     primary version for EXACTLY that one type.
//   - A PARTIAL specialization `TypeName<T*>` (called with "T*") still
//     depends on another type T, and can delegate back to it.
//   - Function templates may not be partially specialized — for a pointer
//     rule you use a class template, as done here.
//
// EXPECTED OUTPUT:
//   TypeName<int>            : int
//   TypeName<double>         : double
//   TypeName<char>           : unknown
//   TypeName<int*>           : pointer to int
//   TypeName<double**>       : pointer to pointer to double

#include <iostream>
#include <string>

// Primary template: the default answer for every type.
template <typename T>
struct TypeName {
    static std::string get() { return "unknown"; }
};

// Full specializations: exactly these concrete types get custom answers.
template <>
struct TypeName<int> {
    static std::string get() { return "int"; }
};

template <>
struct TypeName<double> {
    static std::string get() { return "double"; }
};

// Partial specialization: any pointer type T*. Delegate to TypeName<T>
// so nested pointers work automatically (int** -> "pointer to int*"-...).
template <typename T>
struct TypeName<T*> {
    static std::string get() { return "pointer to " + TypeName<T>::get(); }
};

int main() {
    std::cout << "TypeName<int>            : " << TypeName<int>::get()
              << "\n";
    std::cout << "TypeName<double>         : " << TypeName<double>::get()
              << "\n";
    std::cout << "TypeName<char>           : " << TypeName<char>::get()
              << "\n";
    std::cout << "TypeName<int*>           : " << TypeName<int*>::get()
              << "\n";
    std::cout << "TypeName<double**>       : " << TypeName<double**>::get()
              << "\n";
    return 0;
}