// 01_basics/08_user_input.cpp
// Reading input from the user (std::cin) - and from piped input in tests.
//
// WHAT YOU LEARN:
//   - std::cin >> var   reads one "token" (word/number), stops at whitespace.
//   - std::getline(...) reads everything up to and including a newline.
//   - Mixing >> and getline leaves a leftover newline => classic bug.
//   - Checking "if (!std::cin)" detects invalid input (non-numeric, EOF).
//   - You can loop until the user types a valid answer (like a quiz).
//
// TEST PROTOCOL: to keep this automated, run with piped input, e.g.:
//   echo "Alice`n42`nthe matrix is cool`n25" | .\build\01_basics\conc_user_input.exe
// (In PowerShell use backtick-n inside the string to emit newlines.)
//
// EXPECTED OUTPUT (piped input: "Alice" / 42 / "the matrix is cool" / 25):
//   Enter your name: Hello, Alice! Welcome to the C++ course.
//   Enter your favorite number: Alice, 42 doubled is 84
//   Enter one more sentence: You typed: "the matrix is cool"
//   Enter an age between 1 and 120: Your age 25 is saved.

#include <iostream>
#include <string>

int main() {
    // 1) Whole-line input
    std::cout << "Enter your name: ";
    std::string name;
    std::getline(std::cin, name);               // reads "Alice"
    std::cout << "Hello, " << name << "! Welcome to the C++ course.\n";

    // 2) Numeric input
    std::cout << "Enter your favorite number: ";
    int number;
    std::cin >> number;                         // reads 42
    if (!std::cin) {                            // not a valid integer?
        std::cin.clear();                       //  clear the fail flag
        std::cin.ignore(10000, '\n');           //  discard bad tokens
        number = -1;
        std::cout << "That was not a number; using -1 as a stand-in.\n";
    } else {
        std::cin.ignore(10000, '\n');           // eat the leftover newline
    }
    std::cout << name << ", " << number << " doubled is " << 2 * number
              << "\n";

    // 3) The classic pitfall: >> leaves the newline; getline would read ""
    //    immediately. We already consumed it above so getline works.
    std::cout << "Enter one more sentence: ";
    std::string sentence;
    std::getline(std::cin, sentence);           // reads "the matrix is cool"
    std::cout << "You typed: \"" << sentence << "\"\n";

    // 4) Loop until the answer makes sense (a range check)
    std::cout << "Enter an age between 1 and 120: ";
    int age = 0;
    while (std::cin >> age && (age < 1 || age > 120)) {
        std::cout << "Out of range, try again: ";
    }
    std::cout << "Your age " << age << " is saved.\n";

    return 0;
}