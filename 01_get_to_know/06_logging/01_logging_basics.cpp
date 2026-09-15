// 06_logging/01_logging_basics.cpp
// Logging basics: labeled messages with severity levels.
//
// WHAT YOU LEARN:
//   - INFO  = normal progress, so you can follow what the program did.
//   - WARN  = something unusual but not fatal (value clipped, ...).
//   - ERROR = something failed; the program may still recover.
//   - A tiny helper per level keeps messages tidy and consistent.
//
// EXPECTED OUTPUT:
//   [INFO ] program started
//   [INFO ] user typed 2 numbers
//   [WARN ] score 105 clipped to the maximum of 100
//   [ERROR] could not open 'missing.txt'
//   [INFO ] program finished

#include <iostream>
#include <string>

// --- small helpers, one per severity level ---
void log_info(const std::string& text) {
    std::cout << "[INFO ] " << text << "\n";
}

void log_warn(const std::string& text) {
    std::cout << "[WARN ] " << text << "\n";
}

void log_error(const std::string& text) {
    std::cout << "[ERROR] " << text << "\n";
}

int main() {
    // pretend the program is running a tiny scoring job
    log_info("program started");

    int numbers = 2;
    log_info("user typed " + std::to_string(numbers) + " numbers");

    int raw_score = 105;
    if (raw_score > 100) {
        log_warn("score 105 clipped to the maximum of 100");
    }

    log_error("could not open 'missing.txt'");
    log_info("program finished");

    return 0;
}