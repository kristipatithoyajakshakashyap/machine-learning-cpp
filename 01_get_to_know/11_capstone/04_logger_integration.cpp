// 11_capstone/04_logger_integration.cpp
// Putting it together: a tiny end-to-end pipeline with a logger.
//
// WHAT YOU LEARN:
//   - Real programs chain steps: read a file -> compute -> report.
//   - A Logger (from module 06) narrates every step to console and file;
//     the timestamps use a fixed clock here so the output is identical
//     on every machine (08-style trade-off for testable demos).
//   - optional lets the pipeline UNIFY success and failure paths.
//
// EXPECTED OUTPUT:
//   [2026-09-10 14:30:05] [INFO ] pipeline: read dataset.csv, 4 rows
//   [2026-09-10 14:30:05] [INFO ] pipeline: mean height = 171.25
//   [2026-09-10 14:30:05] [WARN ] pipeline: weight span is 22 (narrow)
//   [2026-09-10 14:30:05] [ERROR] pipeline: step 3 has no data
//   [2026-09-10 14:30:05] [INFO ] pipeline: finished (log in capstone.log)

#include <cstdio>     // std::snprintf
#include <ctime>      // std::tm
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

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
    explicit Logger(const std::string& file, Level min)
        : min_(min), file_(OUT_DIR + "/" + file) {}

    void info(const std::string& msg)  { write(kInfo, msg); }
    void warn(const std::string& msg)  { write(kWarn, msg); }
    void error(const std::string& msg) { write(kError, msg); }

    void write(Level lvl, const std::string& msg) {
        if (lvl < min_) {
            return;
        }
        std::tm when{};
        when.tm_year = 2026 - 1900;    // fixed clock for repeatable output
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

// A column of values (the "dataset" of a later phase).
std::optional<double> column_mean(const std::vector<double>& col) {
    if (col.empty()) {
        return std::nullopt;           // "no data" instead of an exception
    }
    double total = 0.0;
    for (double x : col) {
        total += x;
    }
    return total / static_cast<double>(col.size());
}

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
    std::filesystem::remove(OUT_DIR + "/capstone.log");

    Logger log("capstone.log", kInfo);

    // step 1: read a table (04 gave us the reader; here we mock 4 rows)
    std::vector<double> heights{170, 182, 158, 175};
    log.info("pipeline: read dataset.csv, 4 rows");

    // step 2: compute one statistic
    double mean_h = column_mean(heights).value_or(0.0);
    char stat_msg[64];
    std::snprintf(stat_msg, sizeof(stat_msg), "%.2f", mean_h);
    log.info(std::string("pipeline: mean height = ") + stat_msg);

    // step 3: report a believable warning about spread
    double weight_span = 74.0 - 52.0;
    log.warn("pipeline: weight span is " +
             std::to_string(static_cast<int>(weight_span)) + " (narrow)");

    // step 4: show an optional-driven failure path
    std::vector<double> empty;         // a column that read no rows
    if (!column_mean(empty).has_value()) {
        log.error("pipeline: step 3 has no data");
    }

    log.info("pipeline: finished (log in capstone.log)");
    return 0;
}