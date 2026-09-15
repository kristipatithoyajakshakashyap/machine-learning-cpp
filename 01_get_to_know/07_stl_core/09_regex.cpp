// 07_stl_core/09_regex.cpp
// Pattern matching with <regex>: match, search, replace, capture groups,
// iterating over every match, and two tiny parsers (key=value, dates).
//
// WHAT YOU LEARN:
//   - std::regex_match needs the WHOLE string to fit the pattern;
//     std::regex_search finds the pattern ANYWHERE inside it.
//   - Parentheses create capture groups; std::smatch m gives m[0] (whole
//     match) and m[1], m[2], ... (groups). m.prefix()/suffix() are the rest.
//   - std::sregex_iterator walks every non-overlapping match in a string.
//   - std::regex_replace rewrites matches; $1 refers to a group.
//   - The default grammar is ECMAScript (JavaScript-style): \d \w \s, {n},
//     lazy quantifiers *?, no lookbehind, and no named groups. Write the
//     pattern in a raw string R"(...)" so backslashes stay literal.
//   - Build each std::regex ONCE (construction is slow) and reuse it.
//
// EXPECTED OUTPUT:
//   regex_match("2024-01-15", date)  = yes
//   regex_match("on 2024-01-15", date) = no
//   regex_search("on 2024-01-15", date) = yes -> year=2024 month=01 day=15
//   prefix="on " suffix=""
//   emails found: 2 -> alice@example.com bob@mail.org
//   words: 5 -> the quick brown fox jumps
//   replace digits: "room 101, floor 7" -> "room #, floor #"
//   swap name: "Doe, John" -> "John Doe"
//   key=value parse:
//     host = localhost
//     port = 8080
//     debug = true
//   dates (dd/mm/yyyy -> ISO): 05/03/2021 -> 2021-03-05, 31/12/1999 -> 1999-12-31
//   bad date "2021/03/05" rejected: yes

#include <iostream>
#include <regex>
#include <string>
#include <vector>

const char* yn(bool b) { return b ? "yes" : "no"; }

// Parse "key=value; key2=value2" style text into (key, value) pairs.
std::vector<std::pair<std::string, std::string>> parse_kv(const std::string& text) {
    static const std::regex kv(R"((\w+)\s*=\s*([^;\s]+))");
    std::vector<std::pair<std::string, std::string>> out;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), kv);
         it != std::sregex_iterator(); ++it) {
        out.emplace_back((*it)[1].str(), (*it)[2].str());
    }
    return out;
}

// dd/mm/yyyy -> yyyy-mm-dd, or empty string if the text is not that shape.
std::string to_iso(const std::string& dmy) {
    static const std::regex pattern(R"((\d{2})/(\d{2})/(\d{4}))");
    std::smatch m;
    if (!std::regex_match(dmy, m, pattern)) return "";
    return m[3].str() + "-" + m[2].str() + "-" + m[1].str();
}

int main() {
    // --- match vs search + capture groups ---------------------------------
    const std::regex date(R"((\d{4})-(\d{2})-(\d{2}))");
    const std::string exact = "2024-01-15";
    const std::string inside = "on 2024-01-15";
    std::cout << "regex_match(\"" << exact << "\", date)  = "
              << yn(std::regex_match(exact, date)) << '\n';
    std::cout << "regex_match(\"" << inside << "\", date) = "
              << yn(std::regex_match(inside, date)) << '\n';

    std::smatch m;
    const bool found = std::regex_search(inside, m, date);
    std::cout << "regex_search(\"" << inside << "\", date) = " << yn(found);
    if (found) {
        std::cout << " -> year=" << m[1] << " month=" << m[2] << " day=" << m[3] << '\n';
        std::cout << "prefix=\"" << m.prefix() << "\" suffix=\"" << m.suffix() << "\"\n";
    }

    // --- iterating over every match -----------------------------------------
    const std::regex email(R"([\w.]+@[\w.]+\.\w+)");
    const std::string text = "contact alice@example.com or bob@mail.org today";
    std::vector<std::string> emails;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), email);
         it != std::sregex_iterator(); ++it) {
        emails.push_back(it->str());
    }
    std::cout << "emails found: " << emails.size() << " ->";
    for (const auto& e : emails) std::cout << ' ' << e;
    std::cout << '\n';

    const std::regex word(R"(\b[a-z]+\b)");
    const std::string sentence = "the quick brown fox jumps";
    std::cout << "words: "
              << std::distance(std::sregex_iterator(sentence.begin(), sentence.end(), word),
                               std::sregex_iterator())
              << " ->";
    for (auto it = std::sregex_iterator(sentence.begin(), sentence.end(), word);
         it != std::sregex_iterator(); ++it) {
        std::cout << ' ' << it->str();
    }
    std::cout << '\n';

    // --- regex_replace, with and without groups ----------------------------
    const std::string rooms = "room 101, floor 7";
    std::cout << "replace digits: \"" << rooms << "\" -> \""
              << std::regex_replace(rooms, std::regex(R"(\d+)"), "#") << "\"\n";
    const std::string name = "Doe, John";
    std::cout << "swap name: \"" << name << "\" -> \""
              << std::regex_replace(name, std::regex(R"((\w+),\s*(\w+))"), "$2 $1")
              << "\"\n";

    // --- tiny parsers --------------------------------------------------------
    std::cout << "key=value parse:\n";
    for (const auto& [k, v] : parse_kv("host=localhost; port = 8080; debug=true")) {
        std::cout << "  " << k << " = " << v << '\n';
    }
    std::cout << "dates (dd/mm/yyyy -> ISO): 05/03/2021 -> " << to_iso("05/03/2021")
              << ", 31/12/1999 -> " << to_iso("31/12/1999") << '\n';
    std::cout << "bad date \"2021/03/05\" rejected: " << yn(to_iso("2021/03/05").empty())
              << '\n';
    return 0;
}
