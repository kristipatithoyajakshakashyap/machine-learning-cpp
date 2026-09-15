// 02_memory_management/01_arrays_vectors.cpp
// Storing many values: C-style arrays and std::vector.
//
// WHAT YOU LEARN:
//   - A C-style array has a FIXED size chosen when you write the code.
//   - std::vector can GROW at runtime (push_back).
//   - .size() always tells you the current length of a vector.
//   - [i] and .at(i) read or write one element.
//   - A 2D array is really "an array of arrays" (like a game grid).
//
// EXPECTED OUTPUT:
//   C-style array: scores[0]=90 scores[4]=79 (fixed size 5, keeps all 5)
//   vector grades: 90 85 88   (size 3 after push_back)
//   grades[1] = 85 ; grades.at(2) = 88
//   vector grades is currently size 3 of capacity 4 (capacity may differ
//   on other compilers)
//   highest score in grades: 90
//   TicTacToe grid:
//    1 0 1
//    0 2 0
//    1 1 0
//   grid cell (row 1, column 2) = 0

#include <iostream>
#include <vector>

int main() {
    // --- C-style array: fixed size ---
    int scores[5] = {90, 85, 88, 92, 79};
    std::cout << "C-style array: scores[0]=" << scores[0] << " scores[4]="
              << scores[4] << " (fixed size 5, keeps all 5)\n";

    // --- std::vector: can grow ---
    std::vector<int> grades = {90, 85};
    grades.push_back(88);                       // grows to 3 elements
    std::cout << "vector grades:";
    for (int g : grades) {
        std::cout << " " << g;
    }
    std::cout << "   (size 3 after push_back)\n";

    std::cout << "grades[1] = " << grades[1] << " ; grades.at(2) = "
              << grades.at(2) << "\n";
    std::cout << "vector grades is currently size " << grades.size()
              << " of capacity " << grades.capacity() << " (capacity may vary)\n";

    // --- find the highest value with a loop ---
    int highest = grades[0];
    for (int g : grades) {
        if (g > highest) {
            highest = g;
        }
    }
    std::cout << "highest score in grades: " << highest << "\n";

    // --- 2D array: a TicTacToe board (0 empty, 1 = X, 2 = O) ---
    int grid[3][3] = {
        {1, 0, 1},
        {0, 2, 0},
        {1, 1, 0},
    };
    std::cout << "TicTacToe grid:\n";
    for (int row = 0; row < 3; ++row) {
        std::cout << " ";
        for (int col = 0; col < 3; ++col) {
            std::cout << grid[row][col];
            if (col < 2) {
                std::cout << " ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "grid cell (row 1, column 2) = " << grid[1][2] << "\n";

    return 0;
}