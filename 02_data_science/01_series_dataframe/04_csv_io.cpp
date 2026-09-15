// 04_csv_io.cpp
// Lesson: read_csv scans the header, decides each column's type from its
// cells (all-numeric -> numeric column with missing cells, otherwise string),
// and write_csv round-trips a DataFrame back to disk.
// Equivalent: pandas.read_csv / DataFrame.to_csv.
//
// Data: iris.csv, tips.csv, flights.csv from the data/ folder (real data).
//
// EXPECTED OUTPUT:
// ## iris.csv loaded
// 150 rows x 5 columns
//    sepal_length  sepal_width  petal_length  petal_width  species
// 0           5.1          3.5           1.4          0.2   setosa
// 1           4.9            3           1.4          0.2   setosa
// 2           4.7          3.2           1.3          0.2   setosa
//
//
// ## tips.csv loaded (quoted cells are handled by the parser)
// 244 rows x 7 columns
//    total_bill   tip     sex  smoker  day    time  size
// 0       16.99  1.01  Female      No  Sun  Dinner     2
// 1       10.34  1.66    Male      No  Sun  Dinner     3
// 2       21.01   3.5    Male      No  Sun  Dinner     3
//
//
// DataFrame with 244 rows x 7 columns:
//   #0 total_bill          float64   missing=0
//   #1 tip                 float64   missing=0
//   #2 sex                 str       missing=0
//   #3 smoker              str       missing=0
//   #4 day                 str       missing=0
//   #5 time                str       missing=0
//   #6 size                float64   missing=0
//
// ## flights.csv
// 144 rows x 3 columns
// DataFrame with 144 rows x 3 columns:
//   #0 year                float64   missing=0
//   #1 month               str       missing=0
//   #2 passengers          float64   missing=0
//
//
// ## <results/04_csv_io_results>/iris_first3.csv written and read back
// 3 rows x 5 columns
//    sepal_length  sepal_width  petal_length  petal_width  species
// 0           5.1          3.5           1.4          0.2   setosa
// 1           4.9            3           1.4          0.2   setosa
// 2           4.7          3.2           1.3          0.2   setosa
//
//
// ## missing file is reported:
//   cannot open file '/no/such/file.csv'

#include <iostream>

#include "dsts/csv.hpp"
#include "dsts/dataframe.hpp"
#include <filesystem>

int main() {
    std::filesystem::create_directories(RUN_OUTPUT_DIR);  // results/<stem>_results/ is created lazily
  // 1. plain dataframe
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");
  std::cout << "## iris.csv loaded\n" << iris.shape() << "\n";
  std::cout << iris.head(3) << "\n\n";

  // 2. a file whose header and string cells are double-quoted
  const dsts::DataFrame tips = dsts::read_csv(DATA_DIR "/tips.csv");
  std::cout << "## tips.csv loaded (quoted cells are handled by the parser)\n"
            << tips.shape() << "\n";
  std::cout << tips.head(3) << "\n\n";
  std::cout << tips.info() << "\n";

  // 3. type inference drives info(); passengers is numeric, month is string
  const dsts::DataFrame flights = dsts::read_csv(DATA_DIR "/flights.csv");
  std::cout << "## flights.csv\n" << flights.shape() << "\n";
  std::cout << flights.info() << "\n\n";

  // 4. write a subset, then read it back
  const std::string out_path = std::string(RUN_OUTPUT_DIR) + "/iris_first3.csv";
  const dsts::DataFrame first3 = iris.select_rows({0, 1, 2});
  dsts::write_csv(out_path, first3);
  const dsts::DataFrame reloaded = dsts::read_csv(out_path);
  std::cout << "## " << out_path << " written and read back\n"
            << reloaded.shape() << "\n"
            << reloaded.head(3) << "\n\n";

  // 5. what happens with a file that does not exist
  try {
    (void)dsts::read_csv("/no/such/file.csv");
    std::cout << "read_csv succeeded (unexpected!)\n";
  } catch (const std::runtime_error& e) {
    std::cout << "## missing file is reported:\n  " << e.what() << "\n";
  }

  return 0;
}