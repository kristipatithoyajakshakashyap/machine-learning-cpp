// 06_string_columns.cpp
// Lesson: text columns hide structure. The Titanic Name column follows
// "Surname, Title. Given names (Maiden name)"; a regex pulls surname and
// title apart, trimming and lower-casing make the pieces comparable, and
// rare titles collapse into a handful of categories a model can use.
// Equivalent: df["Name"].str.extract(r",\s*([^\.]+)\.") + value_counts().
//
// Data: titanic.csv (891 passengers, every row has a Name).
//
// EXPECTED OUTPUT:
// parsed 891 names, 891 with a title, 0 unparsed
//
// first 5 rows:
//   Braund, Mr. Owen Harris -> surname=braund title=mr
//   ... (4 more)
//
// raw title counts (17 distinct):
//   mr            517
//   miss          182
//   mrs           125
//   master        40
//   dr            7
//   ... (12 more)
//
// grouped title categories:
//   Mr      517
//   Miss    185   (miss, mlle, ms)
//   Mrs     126   (mrs, mme)
//   Master  40
//   Rare    23   (dr, rev, col, major, capt, don, jonkheer, lady, sir, the countess)
//
// wrote results/06_string_columns_results/titles.csv

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/dataframe.hpp"

namespace {

struct ParsedName {
  std::string surname;
  std::string title;
};

std::string trim(const std::string& s) {
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  const auto b = std::find_if(s.begin(), s.end(), not_space);
  const auto e = std::find_if(s.rbegin(), s.rend(), not_space).base();
  return b < e ? std::string(b, e) : std::string();
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// "Surname, Title. rest" -> {surname, title}; empty strings if no match.
ParsedName parse_name(const std::string& name) {
  static const std::regex pattern(R"(^\s*([^,]+),\s*([^.]+)\.)");
  std::smatch m;
  if (!std::regex_search(name, m, pattern)) return {};
  return {lower(trim(m[1].str())), lower(trim(m[2].str()))};
}

// Collapse raw titles into the five groups used by most Titanic tutorials.
std::string title_group(const std::string& t) {
  if (t == "mr") return "Mr";
  if (t == "miss" || t == "mlle" || t == "ms") return "Miss";
  if (t == "mrs" || t == "mme") return "Mrs";
  if (t == "master") return "Master";
  return "Rare";
}

}  // namespace

int main() {
  const dsts::DataFrame t = dsts::read_csv(DATA_DIR "/titanic.csv");
  const auto& names = t.strings("Name");

  std::vector<ParsedName> parsed;
  parsed.reserve(names.size());
  size_t with_title = 0;
  for (const auto& n : names) {
    parsed.push_back(parse_name(n));
    if (!parsed.back().title.empty()) ++with_title;
  }
  std::cout << "parsed " << names.size() << " names, " << with_title
            << " with a title, " << names.size() - with_title << " unparsed\n\n";

  std::cout << "first 5 rows:\n";
  for (size_t r = 0; r < 5 && r < names.size(); ++r) {
    std::cout << "  " << names[r] << " -> surname=" << parsed[r].surname
              << " title=" << parsed[r].title << "\n";
  }

  // Raw count table, sorted by count descending then name.
  std::map<std::string, size_t> raw_counts;
  for (const auto& p : parsed) ++raw_counts[p.title];
  std::vector<std::pair<std::string, size_t>> ordered(raw_counts.begin(), raw_counts.end());
  std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  });
  std::cout << "\nraw title counts (" << ordered.size() << " distinct):\n";
  for (const auto& [title, count] : ordered) {
    std::cout << "  " << title << std::string(title.size() < 14 ? 14 - title.size() : 1, ' ')
              << count << "\n";
  }

  // Grouped categories with the raw titles that fed each one.
  std::map<std::string, size_t> group_counts;
  std::map<std::string, std::vector<std::string>> group_members;
  for (const auto& [title, count] : ordered) {
    const std::string g = title_group(title);
    group_counts[g] += count;
    group_members[g].push_back(title);
  }
  const std::vector<std::string> group_order{"Mr", "Miss", "Mrs", "Master", "Rare"};
  std::cout << "\ngrouped title categories:\n";
  for (const auto& g : group_order) {
    std::cout << "  " << g << std::string(8 - g.size(), ' ') << group_counts[g];
    const auto& members = group_members[g];
    if (members.size() > 1) {
      std::cout << "   (";
      for (size_t i = 0; i < members.size(); ++i) std::cout << (i ? ", " : "") << members[i];
      std::cout << ")";
    }
    std::cout << "\n";
  }

  // Write the count table: one row per raw title with its group.
  std::filesystem::create_directories(RUN_OUTPUT_DIR);
  dsts::DataFrame out;
  std::vector<std::string> col_title, col_group;
  std::vector<dsts::OptD> col_count;
  for (const auto& [title, count] : ordered) {
    col_title.push_back(title);
    col_group.push_back(title_group(title));
    col_count.push_back(static_cast<double>(count));
  }
  out.add_string("title", col_title);
  out.add_numeric("count", col_count);
  out.add_string("group", col_group);
  const std::string path = std::string(RUN_OUTPUT_DIR) + "/titles.csv";
  dsts::write_csv(path, out);
  std::cout << "\nwrote results/06_string_columns_results/titles.csv\n";
  return 0;
}
