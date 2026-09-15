// 01_basics/04_loops.cpp
// Loops: for, while, do-while, range-based for.
//
// WHAT YOU LEARN:
//   - for:   run a known number of times.
//   - while: keep going WHILE a condition holds.
//   - do-while: runs at least once.
//   - range-based for: visit every element of a container.
//   - break / continue: jump out / skip an iteration.
//
// EXPECTED OUTPUT:
//   [for] counting 0 1 2 3 4
//   [for] simple addition table
//   1 + 1 = 2    1 + 2 = 3    1 + 3 = 4
//   2 + 1 = 3    2 + 2 = 4    2 + 3 = 5
//   3 + 1 = 4    3 + 2 = 5    3 + 3 = 6
//   [while] countdown 5 4 3 2 1
//   [while] double keeps getting halved 7 times before it drops below 1
//   [do-while] payment 1: owed 6
//   [do-while] payment 2: owed 2
//   [do-while] payment 3: owed -2 (cleared!)
//   [range] sample numbers are: 10 20 30 40
//   [continue] skip multiples of 3: 1 2 4 5 7 8 10
//   sum 1..10 = 55

#include <iostream>
#include <vector>

int main() {
    // --- for loop: counting ---
    std::cout << "[for] counting";
    for (int i = 0; i < 5; ++i) {
        std::cout << " " << i;
    }
    std::cout << "\n";

    // --- nested for loop: a small addition table ---
    std::cout << "[for] simple addition table\n";
    for (int row = 1; row <= 3; ++row) {
        std::cout << " ";
        for (int col = 1; col <= 3; ++col) {
            std::cout << row << " + " << col << " = " << (row + col) << "\t";
        }
        std::cout << "\n";
    }

    // --- while loop: countdown ---
    std::cout << "[while] countdown";
    int n = 5;
    while (n >= 1) {
        std::cout << " " << n;
        --n;
    }
    std::cout << "\n";

    // --- while loop with decimals ---
    // TIP: never test a double against exactly 0.0 - a shrinking value
    // can get stuck at the tiniest positive number and loop forever.
    // Compare against a sensible limit like 1.0 instead.
    double value = 100.0;
    int halvings = 0;
    while (value >= 1.0) {
        value /= 2.0;
        ++halvings;
    }
    std::cout << "[while] double keeps getting halved " << halvings
              << " times before it drops below 1\n";

    // --- do-while: the body always runs at least once ---
    // (pretend you are paying back a debt with 4 dollars per payment)
    int owed = 10;
    int payment = 0;
    do {
        owed -= 4;
        ++payment;
        if (owed > 0) {
            std::cout << "[do-while] payment " << payment << ": owed " << owed
                      << "\n";
        } else {
            std::cout << "[do-while] payment " << payment << ": owed " << owed
                      << " (cleared!)\n";
        }
    } while (owed > 0);

    // --- range-based for: visit every element ---
    std::vector<int> samples{10, 20, 30, 40};
    std::cout << "[range] sample numbers are:";
    for (int s : samples) {
        std::cout << " " << s;
    }
    std::cout << "\n";

    // --- continue: skip unwanted iterations ---
    std::cout << "[continue] skip multiples of 3:";
    for (int i = 1; i <= 10; ++i) {
        if (i % 3 == 0) {
            continue;      // skip 3, 6, 9
        }
        std::cout << " " << i;
    }
    std::cout << "\n";

    // --- accumulate a sum ---
    int total = 0;
    for (int i = 1; i <= 10; ++i) {
        total += i;
    }
    std::cout << "sum 1..10 = " << total << "\n";

    return 0;
}