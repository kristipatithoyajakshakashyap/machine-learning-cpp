// 03_concat.cpp
// Lesson: concat_rows() stacks frames vertically; the columns of every frame
// must line up. The iris file is split by species and then reassembled -
// the round trip must give back exactly 150 rows.
// Equivalent: pandas.concat([df1, df2], axis=0).
//
// Data: iris.csv (real data, 150 flowers).
//
// EXPECTED OUTPUT:
// pieces: 50 + 50 + 50
// rebuilt 150 rows x 5 columns
// first row: setosa / 1.4
// last row: virginica / 5.1

#include <iostream>
#include <string>
#include <vector>

#include "dsts/csv.hpp"
#include "dsts/join.hpp"

std::vector<bool> species_mask(const dsts::DataFrame& df,
                               const std::string& species) {
  std::vector<bool> mask(df.rows());
  for (size_t r = 0; r < df.rows(); ++r) {
    mask[r] = df.strings("species")[r] == species;
  }
  return mask;
}

int main() {
  const dsts::DataFrame iris = dsts::read_csv(DATA_DIR "/iris.csv");

  const dsts::DataFrame setosa = iris.filter(species_mask(iris, "setosa"));
  const dsts::DataFrame versicolor =
      iris.filter(species_mask(iris, "versicolor"));
  const dsts::DataFrame virginica = iris.filter(species_mask(iris, "virginica"));

  std::cout << "pieces: " << setosa.rows() << " + " << versicolor.rows()
            << " + " << virginica.rows() << "\n";

  const dsts::DataFrame rebuilt =
      dsts::concat_rows({setosa, versicolor, virginica});
  std::cout << "rebuilt " << rebuilt.shape() << "\n";
  std::cout << "first row: " << rebuilt.cell(0, "species") << " / "
            << dsts::fmt(*rebuilt.numeric("petal_length")[0]) << "\n";
  std::cout << "last row: " << rebuilt.cell(rebuilt.rows() - 1, "species")
            << " / " << dsts::fmt(*rebuilt.numeric("petal_length")[149])
            << "\n";
  return 0;
}