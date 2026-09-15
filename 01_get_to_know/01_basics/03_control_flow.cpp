// 01_basics/03_control_flow.cpp
// Control flow: if / else if / else and switch.
//
// WHAT YOU LEARN:
//   - Branching lets a program make decisions.
//   - if / else if / else: order matters, first true branch wins.
//   - switch: fast multi-way dispatch on integer-like values.
//   - Best practice: prefer switch when comparing ONE variable
//     against many constant values.
//
// EXPECTED OUTPUT:
//   [if-chain] score 85 -> letter B
//   [if-chain] score 55 -> letter F
//   [switch]   day 3    -> Wednesday
//   [switch]   day 7    -> Sunday
//   [switch]   day 9    -> Unknown day
//   [fizzbuzz] 5 -> Fizz
//   [fizzbuzz] 3 -> Buzz
//   [fizzbuzz] 15 -> FizzBuzz
//   [fizzbuzz] 7 -> 7
//   categorize: 7 is positive, 0 is zero, -5 is negative

#include <iostream>
#include <string>

// --- Classic FizzBuzz: demonstrates branching ---
std::string fizzbuzz(int n) {
    if (n % 15 == 0) return "FizzBuzz";
    if (n % 5 == 0)  return "Fizz";
    if (n % 3 == 0)  return "Buzz";
    return std::to_string(n);
}

// --- A simple decision helper: positive, zero or negative? ---
std::string categorize(int n) {
    if (n > 0) return "positive";
    if (n < 0) return "negative";
    return "zero";
}

int main() {
    // --- if / else if / else ---
    int score = 85;
    char letter;
    if (score >= 90) {
        letter = 'A';
    } else if (score >= 80) {
        letter = 'B';
    } else if (score >= 70) {
        letter = 'C';
    } else if (score >= 60) {
        letter = 'D';
    } else {
        letter = 'F';
    }
    std::cout << "[if-chain] score " << score << " -> letter " << letter
              << "\n";

    score = 55;
    if (score >= 90) {
        letter = 'A';
    } else if (score >= 80) {
        letter = 'B';
    } else if (score >= 70) {
        letter = 'C';
    } else if (score >= 60) {
        letter = 'D';
    } else {
        letter = 'F';
    }
    std::cout << "[if-chain] score " << score << " -> letter " << letter
              << "\n";

    // --- switch ---
    int day = 3;  // 1 = Monday ... 7 = Sunday
    std::string day_name;
    switch (day) {
        case 1: day_name = "Monday";    break;
        case 2: day_name = "Tuesday";   break;
        case 3: day_name = "Wednesday"; break;
        case 4: day_name = "Thursday";  break;
        case 5: day_name = "Friday";    break;
        case 6: day_name = "Saturday";  break;
        case 7: day_name = "Sunday";    break;
        default: day_name = "Unknown day";
    }
    std::cout << "[switch]   day " << day << "    -> " << day_name << "\n";

    day = 7;
    switch (day) {
        case 1: day_name = "Monday";    break;
        case 2: day_name = "Tuesday";   break;
        case 3: day_name = "Wednesday"; break;
        case 4: day_name = "Thursday";  break;
        case 5: day_name = "Friday";    break;
        case 6: day_name = "Saturday";  break;
        case 7: day_name = "Sunday";    break;
        default: day_name = "Unknown day";
    }
    std::cout << "[switch]   day " << day << "    -> " << day_name << "\n";

    day = 9;
    switch (day) {
        case 1: day_name = "Monday";    break;
        case 2: day_name = "Tuesday";   break;
        case 3: day_name = "Wednesday"; break;
        case 4: day_name = "Thursday";  break;
        case 5: day_name = "Friday";    break;
        case 6: day_name = "Saturday";  break;
        case 7: day_name = "Sunday";    break;
        default: day_name = "Unknown day";
    }
    std::cout << "[switch]   day " << day << "    -> " << day_name << "\n";

    // --- Classic FizzBuzz exercise ---
    std::cout << "[fizzbuzz] 5 -> " << fizzbuzz(5) << "\n";
    std::cout << "[fizzbuzz] 3 -> " << fizzbuzz(3) << "\n";
    std::cout << "[fizzbuzz] 15 -> " << fizzbuzz(15) << "\n";
    std::cout << "[fizzbuzz] 7 -> " << fizzbuzz(7) << "\n";

    // --- A simple decision example: is a number positive, zero or negative? ---
    std::cout << "categorize: 7 is " << categorize(7) << ", 0 is "
              << categorize(0) << ", -5 is " << categorize(-5) << "\n";

    return 0;
}