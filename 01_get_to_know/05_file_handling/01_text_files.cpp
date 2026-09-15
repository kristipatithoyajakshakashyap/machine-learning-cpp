// 05_file_handling/01_text_files.cpp
// Text files: writing, reading, appending.
//
// WHAT YOU LEARN:
//   - ofstream writes to a file; ifstream reads from it.
//   - Check that opening worked: "if (!out)" means it failed.
//   - std::getline reads one whole line at a time.
//   - std::ios::app opens a file to ADD at the end.
//   - Files are written into build/runtime so the source tree stays clean
//     (that folder is injected through RUN_OUTPUT_DIR by CMake).
//
// EXPECTED OUTPUT:
//   wrote 3 lines to notes.txt
//     read: line one
//     read: second line
//     read: third and final
//   read 3 lines back from the file
//   after appending one line, the file has 4 lines

#include <fstream>    // ifstream, ofstream
#include <iostream>
#include <string>
#include <filesystem>

const std::string OUT_DIR = RUN_OUTPUT_DIR;

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
    const std::string path = OUT_DIR + "/notes.txt";

    // --- write (overwrites any previous content) ---
    std::ofstream out(path);            // default mode: overwrite
    if (!out) {                         // always check that it opened
        std::cout << "error: could not open notes.txt for writing\n";
        return 1;
    }
    out << "line one\n"
        << "second line\n"
        << "third and final\n";
    out.close();                        // always close when you are done
    std::cout << "wrote 3 lines to notes.txt\n";

    // --- read back with getline ---
    std::ifstream in(path);
    if (!in) {
        std::cout << "error: could not open notes.txt for reading\n";
        return 1;
    }
    std::string line;
    int count = 0;
    while (std::getline(in, line)) {
        ++count;
        std::cout << "    read: " << line << "\n";
    }
    std::cout << "read " << count << " lines back from the file\n";

    // --- append (add WITHOUT deleting what is already there) ---
    std::ofstream app(path, std::ios::app);
    app << "appended line\n";
    app.close();

    std::ifstream in2(path);
    int total = 0;
    while (std::getline(in2, line)) {
        ++total;
    }
    std::cout << "after appending one line, the file has " << total
              << " lines\n";

    return 0;
}