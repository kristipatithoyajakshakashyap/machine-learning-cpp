// 02_data_science/src/datetime.cpp
// dsts calendar utilities: ISO date parsing, leap years, day-number
// conversion (proleptic Gregorian, epoch 1970-01-01), weekday, month
// arithmetic and formatting keys for grouping by month/quarter.
//
// Reads/writes: nothing on disk. Build: part of the dsts static library
// (target dsts); used by 13_datetime lessons. Exercised by ctest -R dsts_regression.
// The day-number algorithms follow Howard Hinnant's "days_from_civil" /
// "civil_from_days" derivations, which are branch-free over 400-year eras.

#include "dsts/datetime.hpp"

#include <cstdio>
#include <string>

namespace dsts {

namespace {

// Parses exactly `width` ASCII digits starting at `pos`; false on failure.
bool read_digits(std::string_view s, size_t pos, size_t width, int& out) {
  if (pos + width > s.size()) return false;
  int value = 0;
  for (size_t i = pos; i < pos + width; ++i) {
    const char c = s[i];
    if (c < '0' || c > '9') return false;
    value = value * 10 + (c - '0');
  }
  out = value;
  return true;
}

// Month in 1..12 and day within that month's length.
bool is_valid(const Date& d) {
  return d.month >= 1 && d.month <= 12 && d.day >= 1 && d.day <= days_in_month(d.year, d.month);
}

}  // namespace

// Gregorian rule: divisible by 4, except centuries unless divisible by 400.
bool is_leap_year(int year) {
  return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
}

// Length of a month; 0 for an invalid month number.
int days_in_month(int year, int month) {
  static const int kDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (month < 1 || month > 12) return 0;
  if (month == 2 && is_leap_year(year)) return 29;
  return kDays[month - 1];
}

// Parse "YYYY-MM-DD" or "YYYY-MM" (day defaults to 1). Any other length,
// separator or non-digit yields nullopt, as does an impossible date.
std::optional<Date> parse_date(std::string_view text) {
  Date d{0, 0, 1};
  if (text.size() != 7 && text.size() != 10) return std::nullopt;
  if (!read_digits(text, 0, 4, d.year) || text[4] != '-' || !read_digits(text, 5, 2, d.month))
    return std::nullopt;
  if (text.size() == 10) {
    if (text[7] != '-' || !read_digits(text, 8, 2, d.day)) return std::nullopt;
  }
  if (!is_valid(d)) return std::nullopt;
  return d;
}

// Days since 1970-01-01 (negative before). Years are shifted so March is
// month 0, which puts the leap day at the end of the year and makes the
// day-of-year formula exact. 146097 = days per 400-year era; 719468 = days
// from 0000-03-01 to 1970-01-01.
long days_from_civil(Date d) {
  const long y = d.year - (d.month <= 2 ? 1 : 0);
  const long era = (y >= 0 ? y : y - 399) / 400;
  const long yoe = y - era * 400;                                        // [0, 399]
  const long m = d.month;
  const long doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d.day - 1;   // [0, 365]
  const long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;                // [0, 146096]
  return era * 146097 + doe - 719468;
}

// Inverse of days_from_civil: recover year, month, day from a day number.
Date civil_from_days(long z) {
  z += 719468;
  const long era = (z >= 0 ? z : z - 146096) / 146097;
  const long doe = z - era * 146097;
  const long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const long y = yoe + era * 400;
  const long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const long mp = (5 * doy + 2) / 153;
  const long d = doy - (153 * mp + 2) / 5 + 1;
  const long m = mp + (mp < 10 ? 3 : -9);
  return Date{static_cast<int>(y + (m <= 2 ? 1 : 0)), static_cast<int>(m), static_cast<int>(d)};
}

// 0 = Sunday .. 6 = Saturday. Negative day numbers are wrapped back into 0..6.
int weekday(Date d) {
  const long days = days_from_civil(d);  // 1970-01-01 was a Thursday (4)
  const long wd = (days + 4) % 7;
  return static_cast<int>(wd < 0 ? wd + 7 : wd);
}

// Shift by `months` (may be negative); the day is clamped to the target
// month's length, e.g. Jan 31 + 1 month -> Feb 28/29.
Date add_months(Date d, int months) {
  const int total = d.year * 12 + (d.month - 1) + months;
  const int year = (total >= 0 ? total : total - 11) / 12;
  const int month = total - year * 12 + 1;
  const int dim = days_in_month(year, month);
  return Date{year, month, d.day > dim ? dim : d.day};
}

// "YYYY-MM-DD" with zero padding.
std::string to_string(Date d) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%04d-%02d-%02d", d.year, d.month, d.day);
  return buf;
}

// "YYYY-MM" grouping key.
std::string month_key(Date d) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%04d-%02d", d.year, d.month);
  return buf;
}

// "YYYY-Qn" grouping key; months 1-3 -> Q1, 4-6 -> Q2, ...
std::string quarter_key(Date d) {
  return std::to_string(d.year) + "-Q" + std::to_string((d.month - 1) / 3 + 1);
}

// English name for a weekday() result; "invalid" outside 0..6.
std::string weekday_name(int wd) {
  static const char* const kNames[7] = {"Sunday",   "Monday", "Tuesday", "Wednesday",
                                        "Thursday", "Friday", "Saturday"};
  return (wd >= 0 && wd < 7) ? kNames[wd] : "invalid";
}

}  // namespace dsts
