// 05_file_handling/03_csv_files.cpp
// CSV files: the most common plain-text table format.
//   name,id,score   <-- header row
//   Ada,101,88      <-- data rows
//
// WHAT YOU LEARN:
//   - CSV = comma-separated values: one row per line, commas split columns.
//   - A tiny split() helper breaks each line into columns.
//   - Skip the header row when you only want the real data.
//   - Parse each numeric column with std::stoi and combine the results.
//   - This is the exact pattern later used to load data tables from disk
//     into containers.
//
// EXPECTED OUTPUT:
//   students.csv written (header + 3 rows)
//   Ada    id=101 score=88
//   Grace   id=102 score=91
//   Linus   id=103 score=84
//   parsed 3 rows
//   average score = 87.667

#include <fstream>
#include <iomanip>    // std::setprecision
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

const std::string OUT_DIR = RUN_OUTPUT_DIR;

// Split "Ada,101,88" into {"Ada", "101", "88"} using the comma as a divider.
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> parts;
    std::string current;
    for (char ch : line) {
        if (ch == ',') {
            parts.push_back(current);
            current.clear();
        } else {
            current += ch;
        }
    }
    parts.push_back(current);      // the last column has no comma after it
    return parts;
}

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
    const std::string path = OUT_DIR + "/students.csv";

    // --- write a small CSV ---
    std::ofstream out(path);
    if (!out) {
        std::cout << "error: could not open students.csv for writing\n";
        return 1;
    }
    out << "name,id,score\n"
        << "Ada,101,88\n"
        << "Grace,102,91\n"
        << "Linus,103,84\n";
    out.close();
    std::cout << "students.csv written (header + 3 rows)\n";

    // --- read and parse ---
    std::ifstream in(path);
    if (!in) {
        std::cout << "error: could not open students.csv for reading\n";
        return 1;
    }

    std::string line;
    std::getline(in, line);        // skip the header row
    int rows = 0;
    int score_sum = 0;
    while (std::getline(in, line)) {
        std::vector<std::string> cols = split(line);   // 3 columns
        int id    = std::stoi(cols[1]);
        int score = std::stoi(cols[2]);
        std::cout << std::left << std::setw(7) << cols[0]
                  << " id=" << id << " score=" << score << "\n";
        score_sum += score;
        ++rows;
    }
    std::cout << "parsed " << rows << " rows\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "average score = "
              << static_cast<double>(score_sum) / rows << "\n";
    return 0;
}