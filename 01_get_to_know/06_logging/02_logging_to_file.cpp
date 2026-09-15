// 06_logging/02_logging_to_file.cpp
// Logging to a file: a permanent record of what the program did.
//
// WHAT YOU LEARN:
//   - A file log survives after the console window is gone.
//   - Open in std::ios::app mode so every run APPENDS, never erases.
//   - A timestamp on every line tells you WHEN something happened.
//   - format_time() turns a std::tm into a readable "Y-M-D H:M:S" string.
//     (In a real program you pass std::time(nullptr); here we use a fixed
//     value so the demo output is the same on every machine.)
//
// EXPECTED OUTPUT:
//   wrote 3 timestamped lines to run.log (append mode)
//   added 2 more lines on top of the old ones
//   final file line count = 5
//   last log line is            finished

#include <cstdio>     // std::snprintf
#include <ctime>      // std::tm
#include <filesystem> // std::filesystem::remove
#include <fstream>
#include <iostream>
#include <string>

const std::string OUT_DIR = RUN_OUTPUT_DIR;

// Format a broken-down clock time as "2026-09-10 14:30:05".
std::string format_time(const std::tm& t) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d",
                  t.tm_year + 1900, t.tm_mon + 1, t.tm_mday,
                  t.tm_hour, t.tm_min, t.tm_sec);
    return std::string(buf);
}

// Append one timestamped line to the log file.
void append_log(const std::string& text) {
    std::ofstream log(OUT_DIR + "/run.log", std::ios::app);   // append mode
    std::tm when{};
    when.tm_year = 2026 - 1900;   // fixed clock for a repeatable demo
    when.tm_mon  = 8;             // September (0-based)
    when.tm_mday = 10;
    when.tm_hour = 14;
    when.tm_min  = 30;
    when.tm_sec  = 5;
    log << format_time(when) << "  " << text << "\n";
}

int count_lines(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        ++n;
    }
    return n;
}

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
    std::filesystem::remove(OUT_DIR + "/run.log");   // fresh start each run

    append_log("[INFO ] starting backup");
    append_log("[INFO ] copying 3 files");
    append_log("[ERROR] could not connect to server");

    std::cout << "wrote 3 timestamped lines to run.log (append mode)\n";

    // run again: old lines must survive thanks to std::ios::app
    append_log("[WARN ] retry scheduled");
    append_log("[INFO ] finished");

    int final        = count_lines(OUT_DIR + "/run.log");
    std::cout << "added 2 more lines on top of the old ones\n";
    std::cout << "final file line count = " << final << "\n";

    // show only the last line (deterministic tail of the file)
    std::ifstream in(OUT_DIR + "/run.log");
    std::string last_line;
    std::string current;
    while (std::getline(in, current)) {
        last_line = current;
    }
    std::cout << "last log line is            "
              << last_line.substr(last_line.find(']') + 2) << "\n";

    return 0;
}