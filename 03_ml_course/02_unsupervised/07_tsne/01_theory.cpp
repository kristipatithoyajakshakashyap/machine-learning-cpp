#include <iostream>
int main() {
  std::cout
      << R"LESSON(t-SNE embeds high-dimensional rows into two or three coordinates so that rows which are close neighbours stay close. Each row gets a Gaussian neighbourhood whose width is tuned to a fixed perplexity, roughly the effective number of neighbours; the embedding uses a heavy-tailed Student-t kernel so that moderately distant rows are pushed apart and clusters separate. The objective is the Kullback-Leibler divergence between the two neighbourhood distributions, minimised by gradient descent from a random start. t-SNE preserves local structure only: cluster sizes, inter-cluster distances and axis directions carry no meaning, the result depends on the seed and the perplexity, and there is no map for new rows. It is a visualisation tool, not a feature extractor.)LESSON"
      << '\n';
}
