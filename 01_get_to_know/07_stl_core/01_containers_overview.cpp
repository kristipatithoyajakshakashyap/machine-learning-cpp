// 07_stl_core/01_containers_overview.cpp
// The Standard Template Library: the ready-made containers you build on.
//
// WHAT YOU LEARN:
//   - std::vector : contiguous memory, fast to append at the BACK.
//   - std::list   : doubly linked nodes, fast inserts ANYWHERE.
//   - std::deque  : double-ended queue, fast at both ends.
//   - std::set    : sorted, unique keys, fast lookup.
//   - std::map    : sorted key -> value lookup table.
//   - Pick the container that fits HOW you use the data.
//
// EXPECTED OUTPUT:
//   std::vector: fast at the back         [10 20 30]
//   std::list:   fast inserts anywhere    [one two three]
//   std::deque:  fast at front and back   [1 5 9]
//   std::set:    sorted and unique        [1 2 3 4 5]
//   std::map:    sorted key -> value pairs   [apple=1 pear=2]
//   sizes: vector 3, list 3, deque 3, set 5, map 2

#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

// Print any container that supports begin()/end().
template <typename Container>
void show(const char* label, const char* note, const Container& c) {
    std::cout << label << note << " [";
    bool first = true;
    for (const auto& x : c) {
        if (!first) {
            std::cout << " ";
        }
        std::cout << x;
        first = false;
    }
    std::cout << "]\n";
}

int main() {
    std::vector<int> v{10, 20, 30};
    std::list<std::string> l{"one", "two", "three"};
    std::deque<int> d{1, 5, 9};
    std::set<int> s{5, 1, 4, 2, 3};          // duplicate 1 removed
    std::map<std::string, int> m{{"apple", 1}, {"pear", 2}};

    show("std::vector: ", "fast at the back        ", v);
    show("std::list:   ", "fast inserts anywhere   ", l);
    show("std::deque:  ", "fast at front and back  ", d);
    show("std::set:    ", "sorted and unique       ", s);

    std::cout << "std::map:    sorted key -> value pairs   [";
    bool first = true;
    for (const auto& kv : m) {          // kv is a std::pair
        if (!first) {
            std::cout << " ";
        }
        std::cout << kv.first << "=" << kv.second;
        first = false;
    }
    std::cout << "]\n";

    std::cout << "sizes: vector " << v.size() << ", list " << l.size()
              << ", deque " << d.size() << ", set " << s.size()
              << ", map " << m.size() << "\n";
    return 0;
}