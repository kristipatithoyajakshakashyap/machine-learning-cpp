// 11_capstone/02_csv_dataset_reader.cpp
// Dataset loading: text file -> rows of numbers in memory.
//
// WHAT YOU LEARN:
//   - A "dataset" here is just a table: columns of measurements, rows of
//     records. A Sample struct holds ONE row of two columns.
//   - The reader splits every line on commas (reusing the 05 trick),
//     skips the header, and fills a Dataset.
//   - Keep the original file untouched by working only in memory.
//
// EXPECTED OUTPUT:
//   dataset.csv written (header + 4 rows)
//   loaded 4 samples
//   row 0: height=170.00 weight=65.00
//   row 1: height=182.00 weight=74.00
//   row 2: height=158.00 weight=52.00
//   row 3: height=175.00 weight=70.00
//   average height = 171.25
//   tallest sample has height 182.00

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

const std::string OUT_DIR = RUN_OUTPUT_DIR;

struct Sample {
    double height;
    double weight;
};

// One table: a pile of row-samples.
struct Dataset {
    std::vector<Sample> rows;
};

// "170,65" -> {"170", "65"}
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
    parts.push_back(current);
    return parts;
}

// Read a CSV file into a Dataset, skipping the header line.
Dataset load_csv(const std::string& path) {
    Dataset data;
    std::ifstream in(path);
    if (!in) {
        return data;                 // empty on failure (lesson keeps it
    }                                // simple; 04_logger reports errors)
    std::string line;
    std::getline(in, line);          // skip header "height,weight"
    while (std::getline(in, line)) {
        std::vector<std::string> cols = split(line);
        data.rows.push_back({std::stod(cols[0]), std::stod(cols[1])});
    }
    return data;
}

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
    const std::string path = OUT_DIR + "/dataset.csv";

    // create a small dataset file so the lesson is self-contained
    {
        std::ofstream out(path);
        out << "height,weight\n"
            << "170,65\n"
            << "182,74\n"
            << "158,52\n"
            << "175,70\n";
    }
    std::cout << "dataset.csv written (header + 4 rows)\n";

    Dataset data = load_csv(path);
    std::cout << "loaded " << data.rows.size() << " samples\n";

    std::cout << std::fixed << std::setprecision(2);
    double height_sum = 0.0;
    double max_height = 0.0;
    for (std::size_t i = 0; i < data.rows.size(); ++i) {
        const Sample& s = data.rows[i];
        std::cout << "row " << i << ": height=" << s.height
                  << " weight=" << s.weight << "\n";
        height_sum += s.height;
        if (s.height > max_height) {
            max_height = s.height;
        }
    }

    std::cout << "average height = "
              << height_sum / static_cast<double>(data.rows.size()) << "\n";
    std::cout << "tallest sample has height " << max_height << "\n";
    return 0;
}