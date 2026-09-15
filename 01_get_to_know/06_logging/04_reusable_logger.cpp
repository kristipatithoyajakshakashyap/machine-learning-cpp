// 06_logging/04_reusable_logger.cpp
// A production-flavored Logger: timestamps, plain-text helper methods,
// console + file output, and a runtime-adjustable level.
//
// WHAT YOU LEARN:
//   - Convenience methods info()/warn()/error() make every call site read
//     like a sentence:  logger.error("file not found").
//   - format_time() renders a std::tm as "2026-09-10 14:30:05". A real
//     program passes the current clock (std::time(nullptr)); this demo
//     uses a fixed time so the output is identical on every machine.
//   - set_min_level() lets you quiet the program without editing code.
//
// EXPECTED OUTPUT:
//   [2026-09-10 14:30:05] [INFO ] loading config.json
//   [2026-09-10 14:30:05] [WARN ] config.json missing, using defaults
//   [2026-09-10 14:30:05] [ERROR] service.unavailable after 3 tries
//   level raised to ERROR only -> the next INFO/WARN lines are hidden
//   [2026-09-10 14:30:05] [ERROR] aborting job step 2
//   (each line was also appended to event.log)

#include <cstdio>     // std::snprintf
#include <ctime>      // std::tm
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

const std::string OUT_DIR = RUN_OUTPUT_DIR;

enum Level { kInfo, kWarn, kError };

std::string format_time(const std::tm& t) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d",
                  t.tm_year + 1900, t.tm_mon + 1, t.tm_mday,
                  t.tm_hour, t.tm_min, t.tm_sec);
    return std::string(buf);
}

class Logger {
public:
    explicit Logger(const std::string& file, Level min_level)
        : min_(min_level), file_(OUT_DIR + "/" + file) {}

    void info(const std::string& msg)  { write(kInfo, msg); }
    void warn(const std::string& msg)  { write(kWarn, msg); }
    void error(const std::string& msg) { write(kError, msg); }

    void set_min_level(Level lvl) { min_ = lvl; }

    void write(Level lvl, const std::string& msg) {
        if (lvl < min_) {
            return;
        }
        std::tm when{};
        when.tm_year = 2026 - 1900;    // fixed clock for a repeatable demo
        when.tm_mon  = 8;
        when.tm_mday = 10;
        when.tm_hour = 14;
        when.tm_min  = 30;
        when.tm_sec  = 5;
        std::string line = "[" + format_time(when) + "] [" + tag(lvl) + "] "
                           + msg;
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
    std::filesystem::remove(OUT_DIR + "/event.log");

    Logger logger("event.log", kInfo);
    logger.info("loading config.json");
    logger.warn("config.json missing, using defaults");
    logger.error("service.unavailable after 3 tries");

    logger.set_min_level(kError);            // quiet everything but errors
    std::cout << "level raised to ERROR only -> the next INFO/WARN lines "
                 "are hidden\n";
    logger.info("heartbeat ok");             // hidden
    logger.warn("run slow");                 // hidden
    logger.error("aborting job step 2");

    std::cout << "(each line was also appended to event.log)\n";
    return 0;
}