// 01_missing_values.cpp
// Lesson: real data usually has holes. A missing numeric cell is
// std::nullopt, an empty string cell is a missing string. We count them,
// drop the rows that contain one, and fill Age with its median.
// Equivalent: df.info(), df.dropna(), df.fillna().
//
// Data: penguins.csv and titanic.csv (real data).
//
// EXPECTED OUTPUT:
// ## penguins.csv: 344 rows x 7 columns
// penguins with any missing cell: 11 rows
// dropna() keeps: 333 rows x 7 columns
//
// ## bill_length_mm: 342 present, 2 missing
// mean while missing present = 43.92193
//
// ## titanic Age: 714/891 present
// mean Age  = 29.69912
// median Age = 28
//
// dropna(Age) keeps 714 of 891 rows
// ## fill Age with its median:
// mean Age after fill = 29.36158
// missing now = 0
//
// ## titanic info()
// DataFrame with 891 rows x 12 columns:
//   #0 PassengerId         float64   missing=0
//   #1 Survived            float64   missing=0
//   #2 Pclass              float64   missing=0
//   #3 Name                str       missing=0
//   #4 Sex                 str       missing=0
//   #5 Age                 float64   missing=177
//   #6 SibSp               float64   missing=0
//   #7 Parch               float64   missing=0
//   #8 Ticket              str       missing=0
//   #9 Fare                float64   missing=0
//   #10 Cabin               str       missing=687
//   #11 Embarked            str       missing=2

#include <iostream>

#include "dsts/cleaning.hpp"
#include "dsts/csv.hpp"
#include "dsts/dataframe.hpp"

int main() {
  const dsts::DataFrame penguins = dsts::read_csv(DATA_DIR "/penguins.csv");
  std::cout << "## penguins.csv: " << penguins.shape() << "\n";

  dsts::DataFrame peng_no_missing = dsts::dropna(penguins);
  std::cout << "penguins with any missing cell: "
            << (penguins.rows() - peng_no_missing.rows()) << " rows\n"
            << "dropna() keeps: " << peng_no_missing.shape() << "\n\n";

  const dsts::Series bill = penguins.series("bill_length_mm");
  std::cout << "## bill_length_mm: " << bill.count() << " present, "
            << bill.missing() << " missing\n"
            << "mean while missing present = " << dsts::fmt(bill.mean()) << "\n\n";

  // Titanic: Age has 177 missing cells; mean only uses present Age.
  const dsts::DataFrame titanic = dsts::read_csv(DATA_DIR "/titanic.csv");
  const dsts::Series age = titanic.series("Age");
  std::cout << "## titanic Age: " << age.count() << "/" << age.size()
            << " present\n"
            << "mean Age  = " << dsts::fmt(age.mean()) << "\n"
            << "median Age = " << dsts::fmt(age.median()) << "\n\n";

  // Drop the 177 incomplete rows, then fill them instead and compare.
  const dsts::DataFrame age_complete = dsts::dropna(titanic, {"Age"});
  std::cout << "dropna(Age) keeps " << age_complete.rows() << " of "
            << titanic.rows() << " rows\n";

  const dsts::Series age_filled = age.fill(age.median());
  std::cout << "## fill Age with its median:\n"
            << "mean Age after fill = " << dsts::fmt(age_filled.mean()) << "\n"
            << "missing now = " << age_filled.missing() << "\n\n";

  // info() already reports missing counts per column.
  std::cout << "## titanic info()\n" << titanic.info();

  return 0;
}