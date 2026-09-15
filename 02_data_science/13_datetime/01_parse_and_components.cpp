// 01_parse_and_components.cpp
// Lesson: turn the text column "1949-01" into Date values, pull out the
// components (year, month, weekday of the 1st), spot leap years, and count
// how many rows fall in each year.
// Equivalent: pd.to_datetime(df.Month), .dt.year, .dt.dayofweek,
// .dt.is_leap_year, .groupby(year).size().
//
// Data: air_passengers.csv (real data, 144 monthly totals, 1949-1960).
//
// EXPECTED OUTPUT:
// parsed 144 of 144 rows; unparsable 0
// first 1949-01-01 (Saturday)  last 1960-12-01 (Thursday)
// span 4352 days = 143 months
// leap years in range: 1952 1956 1960
// rows per year: 1949:12 1950:12 ... 1960:12
// (plus a components table for the first six rows)

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/datetime.hpp"

int main() {
  const dsts::DataFrame ap = dsts::read_csv(DATA_DIR "/air_passengers.csv");
  const std::vector<std::string>& month_text = ap.strings("Month");

  std::vector<dsts::Date> dates;
  size_t bad = 0;
  for (const std::string& t : month_text) {
    const std::optional<dsts::Date> d = dsts::parse_date(t);
    if (d) dates.push_back(*d); else ++bad;
  }
  std::cout << "parsed " << dates.size() << " of " << month_text.size() << " rows; unparsable "
            << bad << "\n";
  if (dates.empty()) return 1;

  const dsts::Date first = dates.front(), last = dates.back();
  std::cout << "first " << dsts::to_string(first) << " (" << dsts::weekday_name(dsts::weekday(first))
            << ")  last " << dsts::to_string(last) << " ("
            << dsts::weekday_name(dsts::weekday(last)) << ")\n";
  const long span = dsts::days_from_civil(last) - dsts::days_from_civil(first);
  const int months = (last.year - first.year) * 12 + (last.month - first.month);
  std::cout << "span " << span << " days = " << months << " months\n";

  std::cout << "leap years in range:";
  for (int y = first.year; y <= last.year; ++y)
    if (dsts::is_leap_year(y)) std::cout << ' ' << y;
  std::cout << "\n";

  std::map<int, int> per_year;  // ordered by year
  for (const dsts::Date& d : dates) ++per_year[d.year];
  std::cout << "rows per year:";
  for (const auto& [year, n] : per_year) std::cout << ' ' << year << ':' << n;
  std::cout << "\n\n";

  std::cout << "components of the first six rows\n"
            << "  text     year  month  day  weekday    quarter  +1 month\n";
  for (size_t i = 0; i < 6 && i < dates.size(); ++i) {
    const dsts::Date d = dates[i];
    std::cout << "  " << month_text[i] << "  " << d.year << "  " << d.month << (d.month < 10 ? "      " : "     ")
              << d.day << "    " << dsts::weekday_name(dsts::weekday(d))
              << std::string(11 - dsts::weekday_name(dsts::weekday(d)).size(), ' ')
              << dsts::quarter_key(d) << "  " << dsts::to_string(dsts::add_months(d, 1)) << "\n";
  }

  // Edge cases worth knowing: invalid dates give nullopt, add_months clamps.
  std::cout << "\nparse_date(\"2023-02-30\") valid? " << (dsts::parse_date("2023-02-30") ? "yes" : "no")
            << "   parse_date(\"2024-02-29\") valid? " << (dsts::parse_date("2024-02-29") ? "yes" : "no")
            << "\nadd_months(2024-01-31, 1) = " << dsts::to_string(dsts::add_months({2024, 1, 31}, 1))
            << "  (day clamped to 29)\n";
  return 0;
}
