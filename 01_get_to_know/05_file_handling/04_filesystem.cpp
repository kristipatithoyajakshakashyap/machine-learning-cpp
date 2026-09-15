// 05_file_handling/04_filesystem.cpp
// std::filesystem: working with paths, folders, and file properties.
//
// WHAT YOU LEARN:
//   - std::filesystem::path models a file or folder location.
//   - exists() and is_directory() inspect what is actually on disk.
//   - create_directories() makes whole folders (including parents).
//   - directory_iterator() walks the files inside a folder.
//   - file_size(), rename(), and remove() do the rest of the chores.
//
// EXPECTED OUTPUT:
//   demo_dir exists? true
//   demo_dir is a directory? true
//   entries inside demo_dir:
//     a.txt
//     b.txt
//     sub (folder)
//   a.txt has size 6 bytes     (it holds exactly "hello\n")
//   renamed b.txt to renamed.txt -> exists now? true
//   removed renamed.txt -> still there? false
//   remove_all(demo_dir) deleted 3 entries -> demo_dir exists? false

#include <algorithm>   // std::sort
#include <filesystem>  // std::filesystem
#include <fstream>     // ofstream
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;   // short, friendly alias

const std::string OUT_DIR = RUN_OUTPUT_DIR;

int main() {
    fs::path base = fs::path(OUT_DIR) / "demo_dir";
    fs::create_directories(base / "sub");      // makes demo_dir AND sub

    // --- create two tiny files with raw ofstream ---
    auto write_file = [](const fs::path& p, const std::string& content) {
        std::ofstream f(p, std::ios::binary);   // binary: byte-for-byte exact
        f << content;
    };
    write_file(base / "a.txt", "hello\n");
    write_file(base / "b.txt", "beta data\n");

    // --- what kind of thing is this? ---
    std::cout << "demo_dir exists? " << std::boolalpha << fs::exists(base)
              << "\n";
    std::cout << "demo_dir is a directory? " << fs::is_directory(base)
              << "\n";

    // --- list the folder (sorted for a stable order) ---
    std::cout << "entries inside demo_dir:\n";
    std::vector<fs::path> entries;
    for (const auto& entry : fs::directory_iterator(base)) {
        entries.push_back(entry.path());
    }
    std::sort(entries.begin(), entries.end());   // stable order for printing
    for (const auto& p : entries) {
        std::cout << "    " << p.filename().string()
                  << (fs::is_directory(p) ? " (folder)" : "") << "\n";
    }

    // --- file size / rename / remove ---
    std::cout << "a.txt has size " << fs::file_size(base / "a.txt")
              << " bytes     (it holds exactly \"hello\\n\")\n";

    fs::path moved = base / "renamed.txt";
    fs::rename(base / "b.txt", moved);
    std::cout << "renamed b.txt to renamed.txt -> exists now? "
              << fs::exists(moved) << "\n";

    fs::remove(moved);
    std::cout << "removed renamed.txt -> still there? "
              << fs::exists(moved) << "\n";

    // --- remove_all(): delete a whole tree (demo_dir, sub/, a.txt) in one call ---
    const auto removed_count = fs::remove_all(base);
    std::cout << "remove_all(demo_dir) deleted " << removed_count
              << " entries -> demo_dir exists? " << fs::exists(base) << "\n";

    return 0;
}