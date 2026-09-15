// 04_error_handling/04_failures_and_status.cpp
// Two philosophies for handling problems:
//   - STATUS CODES (return bool + fill a result): for EXPECTED problems
//     that normal users could cause (bad text, missing file, ...).
//   - EXCEPTIONS (throw/catch): for UNEXPECTED problems that should
//     never happen if everything else is correct.
//
// WHAT YOU LEARN:
//   - A function can report success and write a result through an output
//     parameter (int& result).
//   - "12xy" is text that starts with a number but is not a number:
//     std::stoi stops early, so we check how much it consumed.
//   - Real projects mix both styles - pick based on WHO can cause the
//     problem and HOW COMMON it is.
//
// EXPECTED OUTPUT:
//   parse_int("123")  -> ok, value 123
//   parse_int("abc")  -> rejected (returns false)
//   parse_int("12xy") -> rejected (trailing characters)
//   status codes suit EXPECTED problems; exceptions suit UNEXPECTED ones

#include <iostream>
#include <string>

// --- returns true if text is a whole number, then fills result ---
bool parse_int(const std::string& text, int& result) {
    try {
        size_t used = 0;
        result = std::stoi(text, &used);   // try to read a number
        if (used != text.size()) {         // leftover text like "xy"?
            result = 0;
            return false;
        }
        return true;
    } catch (const std::exception&) {      // not a number at all
        return false;
    }
}

int main() {
    int value = 0;

    if (parse_int("123", value)) {
        std::cout << "parse_int(\"123\")  -> ok, value " << value << "\n";
    } else {
        std::cout << "parse_int(\"123\")  -> rejected\n";
    }

    if (parse_int("abc", value)) {
        std::cout << "parse_int(\"abc\")  -> ok, value " << value << "\n";
    } else {
        std::cout << "parse_int(\"abc\")  -> rejected (returns false)\n";
    }

    if (parse_int("12xy", value)) {
        std::cout << "parse_int(\"12xy\") -> ok, value " << value << "\n";
    } else {
        std::cout << "parse_int(\"12xy\") -> rejected "
                     "(trailing characters)\n";
    }

    std::cout << "status codes suit EXPECTED problems; "
                 "exceptions suit UNEXPECTED ones\n";
    return 0;
}