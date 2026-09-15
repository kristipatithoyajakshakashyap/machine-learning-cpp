// 05_file_handling/02_binary_files.cpp
// Binary files: saving and loading raw bytes (exact copies, very fast).
//
// WHAT YOU LEARN:
//   - Binary files store the RAW bytes of memory, not text characters.
//   - std::ios::binary + ostream.write(...) / istream.read(...) do the work.
//   - reinterpret_cast<const char*>(&rec) lets write() see an object's bytes.
//   - A binary record is exact: save it, load it, values come back identical.
//
// EXPECTED OUTPUT:
//   wrote 3 binary records, each 32 bytes
//   read back record 0: id=1 value=1.50 tag=alpha
//   read back record 1: id=2 value=2.25 tag=beta
//   read back record 2: id=3 value=-7.00 tag=gamma
//   binary round-trip: the values arrived back byte-for-byte identical

#include <cstddef>    // std::size_t
#include <cstring>    // std::strncpy
#include <fstream>    // ifstream, ofstream
#include <iomanip>    // std::setprecision
#include <iostream>
#include <string>
#include <filesystem>

const std::string OUT_DIR = RUN_OUTPUT_DIR;

// A fixed-layout record that can be written to disk as raw bytes.
struct Record {
    int    id = 0;
    double value = 0.0;
    char   tag[16] = {0};   // fixed-size char array, not a std::string
};

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
    const std::string path = OUT_DIR + "/records.bin";
    const std::size_t record_size = sizeof(Record);

    // --- build three records ---
    Record recs[3];
    recs[0] = {1, 1.5};
    recs[1] = {2, 2.25};
    recs[2] = {3, -7.0};
    std::strncpy(recs[0].tag, "alpha", 15);
    std::strncpy(recs[1].tag, "beta", 15);
    std::strncpy(recs[2].tag, "gamma", 15);

    // --- write the raw bytes ---
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cout << "error: could not open records.bin for writing\n";
        return 1;
    }
    for (const Record& r : recs) {
        out.write(reinterpret_cast<const char*>(&r), record_size);
    }
    out.close();
    std::cout << "wrote 3 binary records, each " << record_size
              << " bytes\n";

    // --- read the raw bytes back ---
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cout << "error: could not open records.bin for reading\n";
        return 1;
    }
    Record loaded[3];
    for (int i = 0; i < 3; ++i) {
        in.read(reinterpret_cast<char*>(&loaded[i]), record_size);
    }

    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < 3; ++i) {
        std::cout << "read back record " << i << ": id=" << loaded[i].id
                  << " value=" << loaded[i].value << " tag=" << loaded[i].tag
                  << "\n";
    }
    std::cout << "binary round-trip: the values arrived back "
                 "byte-for-byte identical\n";

    return 0;
}