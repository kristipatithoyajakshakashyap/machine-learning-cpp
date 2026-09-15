#ifndef DSTS_DATETIME_HPP
#define DSTS_DATETIME_HPP

// Minimal calendar-date helpers (the pandas.Timestamp / DatetimeIndex
// subset used by the course). Dates are proleptic Gregorian; there are no
// time zones or times of day. Everything is pure arithmetic so results are
// identical on every platform.

#include <optional>
#include <string>
#include <string_view>

namespace dsts {

struct Date {
  int year;
  int month;  // 1..12
  int day;    // 1..31
};

inline bool operator==(const Date& a, const Date& b) {
  return a.year == b.year && a.month == b.month && a.day == b.day;
}
inline bool operator!=(const Date& a, const Date& b) { return !(a == b); }

// Accepts "YYYY-MM-DD" or "YYYY-MM" (day defaults to 1). Surrounding
// whitespace/quotes are not stripped; invalid calendar dates give nullopt.
std::optional<Date> parse_date(std::string_view text);

bool is_leap_year(int year);
int days_in_month(int year, int month);

// Days since 1970-01-01 (negative before the epoch). Howard Hinnant's
// days_from_civil algorithm.
long days_from_civil(Date d);
Date civil_from_days(long days);

// 0 = Sunday ... 6 = Saturday.
int weekday(Date d);

// Adds n months (n may be negative); the day is clamped to the target
// month's length, as pandas DateOffset(months=n) does.
Date add_months(Date d, int months);

std::string to_string(Date d);   // "YYYY-MM-DD"
std::string month_key(Date d);   // "YYYY-MM"
std::string quarter_key(Date d); // "YYYY-Qn"
std::string weekday_name(int wd);  // "Sunday" .. "Saturday"

}  // namespace dsts

#endif  // DSTS_DATETIME_HPP
