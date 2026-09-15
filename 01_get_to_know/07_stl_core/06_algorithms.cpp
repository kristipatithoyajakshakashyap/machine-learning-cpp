// 07_stl_core/06_algorithms.cpp
// Algorithm headers: ready-made logic for sorting, searching, counting.
//
// WHAT YOU LEARN:
//   - std::sort(begin, end)            sorts any random-access range.
//   - std::min_element / max_element   scan for the extremes.
//   - std::count_if(begin, end, pred)  counts elements passing a test
//     (the predicate is a small lambda: [](int x){ return x % 2 == 0; }).
//   - std::accumulate(begin, end, init) adds everything together.
//   - std::find(begin, end, value)     searches for one value.
//   - std::transform(begin, end, out, f) maps every element through f.
//
// EXPECTED OUTPUT:
//   original           : 4 1 6 3 2
//   sorted             : 1 2 3 4 6
//   min and max        : min=1 max=6
//   count_if even      : 3
//   accumulate sum     : 16
//   find 3 at index    : 2
//   transform *2       : 2 4 6 8 12

#include <algorithm>   // sort, min_element, count_if, find, transform
#include <iostream>
#include <numeric>     // accumulate
#include <vector>

void print(const char* label, const std::vector<int>& v) {
    std::cout << label << ": ";
    for (int x : v) {
        std::cout << x << " ";
    }
    std::cout << "\n";
}

int main() {
    std::vector<int> v{4, 1, 6, 3, 2};
    print("original           ", v);

    std::sort(v.begin(), v.end());
    print("sorted             ", v);

    int lo = *std::min_element(v.begin(), v.end());
    int hi = *std::max_element(v.begin(), v.end());
    std::cout << "min and max        : min=" << lo << " max=" << hi << "\n";

    int evens = std::count_if(v.begin(), v.end(),
                              [](int x) { return x % 2 == 0; });
    std::cout << "count_if even      : " << evens << "\n";

    int total = std::accumulate(v.begin(), v.end(), 0);
    std::cout << "accumulate sum     : " << total << "\n";

    auto pos = std::find(v.begin(), v.end(), 3);
    std::cout << "find 3 at index    : " << pos - v.begin() << "\n";

    std::vector<int> doubled(v.size());
    std::transform(v.begin(), v.end(), doubled.begin(),
                   [](int x) { return x * 2; });
    print("transform *2       ", doubled);

    return 0;
}