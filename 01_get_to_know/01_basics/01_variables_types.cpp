// 01_basics/01_variables_types.cpp
// Variables and fundamental data types.
//
// WHAT YOU LEARN:
//   - Built-in types: bool, char, int, long, double, float, auto.
//   - sizeof() tells you how many BYTES a type occupies in memory.
//   - auto lets the compiler guess the type from the value you write.
//   - Division behaves differently for whole numbers vs decimals.
//
// EXPECTED OUTPUT:
//   bool    : 1 bytes
//   daytime? true  raining? false
//   char    : 1 bytes
//   a grade letter is stored as a char: A
//   int     : 4 bytes
//   long    : 4 bytes  (varies by platform)
//   an int can hold a count like 150
//   float   : 4 bytes
//   double  : 8 bytes
//   pi_float holds 3.14, pi_double 3.14
//   auto guessed: 3.14 is a double (costs 8 bytes)
//   Integer division 7 / 2 = 3  (fractional part is lost!)
//   Double division  6.0 / 2.0 = 3
//   Casted mean of 7 and 2 = 3.5
//   precision note: float gave   3.14159274
//                   double gave 3.14159265
//   an app with 285000 downloads -> stored as 285000

#include <iomanip>       // std::setprecision
#include <iostream>
#include <type_traits>   // std::is_same_v

int main() {
    std::cout << std::boolalpha;   // print bools as true/false, not 1/0

    // --- Booleans: true or false ---
    bool is_daytime = true;
    bool is_raining = false;
    std::cout << "bool    : " << sizeof(bool) << " bytes\n";
    std::cout << "daytime? " << is_daytime << "  raining? " << is_raining
              << "\n";

    // --- Characters: one symbol ---
    char grade = 'A';
    std::cout << "char    : " << sizeof(char) << " bytes\n";
    std::cout << "a grade letter is stored as a char: " << grade << "\n";

    // --- Integers: whole numbers ---
    int students = 150;
    std::cout << "int     : " << sizeof(int) << " bytes\n";
    std::cout << "long    : " << sizeof(long)
              << " bytes  (varies by platform)\n";
    std::cout << "an int can hold a count like " << students << "\n";

    // --- Floating point: numbers with decimals ---
    double pi_double = 3.14;
    float  pi_float  = 3.14f;               // 'f' suffix => float
    std::cout << "float   : " << sizeof(float) << " bytes\n";
    std::cout << "double  : " << sizeof(double) << " bytes\n";
    std::cout << "pi_float holds " << pi_float << ", pi_double " << pi_double
              << "\n";

    // --- auto: let the compiler guess the type ---
    auto guessed = 3.14;                    // a decimal number => double
    std::cout << "auto guessed: 3.14 is a "
              << (std::is_same_v<decltype(guessed), double> ? "double"
                                                            : "not a double")
              << " (costs " << sizeof(guessed) << " bytes)\n";

    // --- Integer division trap ---
    int a = 7;
    int b = 2;
    std::cout << "Integer division 7 / 2 = " << a / b
              << "  (fractional part is lost!)\n";
    std::cout << "Double division  6.0 / 2.0 = " << 6.0 / 2.0 << "\n";

    // --- Casting: turn one type into another ---
    double mean = static_cast<double>(a) / b;  // force decimal division
    std::cout << "Casted mean of 7 and 2 = " << mean << "\n";

    // --- Precision: float vs double for 3.14159265 ---
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "precision note: float gave   "
              << static_cast<float>(3.14159265) << "\n"
              << "                double gave " << 3.14159265 << "\n";

    // --- A big whole number ---
    long app_downloads = 285000;
    std::cout << "an app with 285000 downloads -> stored as " << app_downloads
              << "\n";

    return 0;
}