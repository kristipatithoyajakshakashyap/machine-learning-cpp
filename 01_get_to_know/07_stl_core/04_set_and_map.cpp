// 07_stl_core/04_set_and_map.cpp
// set and map: sorted, fast, deduplicated lookups.
//
// WHAT YOU LEARN:
//   - std::set<Key>  : keeps keys SORTED and strips duplicates.
//     insert/contains/count are all fast (logarithmic).
//   - std::map<Key,V>: one value per key, keys kept sorted.
//     marks["ada"] = 95 both writes and updates.
//     find() returns an iterator or end(); operator[] never throws.
//   - Iterating gives keys in ascending order for free.
//
// EXPECTED OUTPUT:
//   set after inserts (dupe 3 removed): 1 3 5 7 9
//   set contains 3? count=1   contains 6? count=0
//   map sorted by key: ada=92 bob=88 carol=77
//   after marks["ada"]=95 :  ada=95
//   find bob -> bob has 88

#include <iostream>
#include <map>
#include <set>
#include <string>

int main() {
    // --- set: sorted + unique ---
    std::set<int> ids{7, 3, 3, 9, 1};         // duplicate 3 merged
    ids.insert(5);
    std::cout << "set after inserts (dupe 3 removed): ";
    for (int id : ids) {
        std::cout << id << " ";
    }
    std::cout << "\n";
    std::cout << "set contains 3? count=" << ids.count(3)
              << "   contains 6? count=" << ids.count(6) << "\n";

    // --- map: key -> value, sorted by key ---
    std::map<std::string, int> marks{{"bob", 88}, {"ada", 92},
                                     {"carol", 77}};
    std::cout << "map sorted by key: ";
    bool first = true;
    for (const auto& kv : marks) {
        if (!first) {
            std::cout << " ";
        }
        std::cout << kv.first << "=" << kv.second;
        first = false;
    }
    std::cout << "\n";

    marks["ada"] = 95;                        // update existing key
    std::cout << "after marks[\"ada\"]=95 :  ada=" << marks["ada"] << "\n";

    auto it = marks.find("bob");
    if (it != marks.end()) {
        std::cout << "find bob -> bob has " << it->second << "\n";
    }
    return 0;
}