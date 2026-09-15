// 01_basics/07_strings.cpp
// Text with std::string: joining, measuring, searching, converting.
//
// WHAT YOU LEARN:
//   - Strings can be glued together with +.
//   - .size() gives the length in characters.
//   - [index] and .at(index) read one character at a time.
//   - .find() searches for a smaller piece of text.
//   - .substr() cuts out part of a string.
//   - std::to_string / std::stod / std::stoi convert numbers to text and back.
//
// Expected first characters: 'H'
// 'under' is at position 11 in 'The cat is under the table'
// EXPECTED OUTPUT:
//   full = Hello C++ world!
//   length of full = 16
//   first character of 'Hello' is H
//   letter at position 3 is 'l'
//   'under' found at position 11
//   middle part of 'good morning' = 'morning'
//   to_string(42) = "42"
//   to_string(3.14) = "3.140000"   (to_string always writes 6 decimals)
//   stod("3.14") = 3.14
//   'apple' == 'apple' is true
//   an empty string has size 0 (empty == true)

#include <iostream>
#include <string>

int main() {
    // --- building a string ---
    std::string first = "Hello ";
    std::string second = "C++";
    std::string full = first + second + " world!";
    std::cout << "full = " << full << "\n";

    // --- length ---
    std::cout << "length of full = " << full.size() << "\n";

    // --- reading one character ---
    std::string word = "Hello";
    std::cout << "first character of 'Hello' is " << word[0] << "\n";
    std::cout << "letter at position 3 is " << word[3] << "\n";

    // --- searching ---
    std::string sentence = "The cat is under the table";
    size_t pos = sentence.find("under");
    std::cout << "'under' found at position " << pos << "\n";

    // --- cutting out a piece ---
    std::string greeting = "good morning";
    std::string middle = greeting.substr(5);   // from position 5 to the end
    std::cout << "middle part of 'good morning' = '" << middle << "'\n";

    // --- numbers <-> text ---
    std::cout << "to_string(42) = \"" << std::to_string(42) << "\"\n";
    std::cout << "to_string(3.14) = \"" << std::to_string(3.14) << "\"\n";
    double pi = std::stod("3.14");
    std::cout << "stod(\"3.14\") = " << pi << "\n";

    // --- comparing ---
    std::string a = "apple";
    std::string b = "apple";
    std::cout << "'apple' == 'apple' is " << std::boolalpha << (a == b)
              << "\n";

    // --- empty check ---
    std::string nothing;
    std::cout << "an empty string has size " << nothing.size()
              << " (empty == " << nothing.empty() << ")\n";

    return 0;
}