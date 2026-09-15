// 06_logging/03_multichannel_logger.cpp
// One Logger class that writes to BOTH console and file, with a
// minimum-level filter so you only see what matters.
//
// WHAT YOU LEARN:
//   - A class can hold an ofstream and reuse it for every message.
//   - The enum Level names the severities; comparing ints filters them.
//   - Every message goes to two sinks (stdout + file) with one call.
//
// EXPECTED OUTPUT:
//   [WARN ] disk space below 10%
//   [ERROR] could not open config file
//   results also written to runner.log (INFO lines suppressed,
//   minimum level set to WARN)

#include <filesystem> // std::filesystem::remove
#include <fstream>
#include <iostream>
#include <string>

const std::string OUT_DIR = RUN_OUTPUT_DIR;

enum Level { kInfo, kWarn, kError };

// Reports a message to the console AND to a log file, but only if the
// message is at least as severe as the configured minimum level.
class Log {
public:
    Log(const std::string& file, Level minimum)
        : min_(minimum), file_(OUT_DIR + "/" + file) {}

    void write(Level lvl, const std::string& msg) {
        if (lvl < min_) {
            return;                    // too quiet for this run's config
        }
        std::string line = "[" + tag(lvl) + "] " + msg;
        std::ofstream out(file_, std::ios::app);
        out << line << "\n";
        std::cout << line << "\n";
    }

private:
    static std::string tag(Level lvl) {
        switch (lvl) {
            case kInfo:  return "INFO ";
            case kWarn:  return "WARN ";
            case kError: return "ERROR";
        }
        return "?????";
    }

    Level       min_;
    std::string file_;
};

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
    std::filesystem::remove(OUT_DIR + "/runner.log");

    Log logger("runner.log", kWarn);   // filter out INFO
    logger.write(kInfo, "starting job runner");       // suppressed
    logger.write(kWarn, "disk space below 10%");
    logger.write(kError, "could not open config file");

    std::cout << "results also written to runner.log (INFO lines "
                 "suppressed,\n"
                 "minimum level set to WARN)\n";
    return 0;
}