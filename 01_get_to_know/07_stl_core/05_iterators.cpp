// 07_stl_core/05_iterators.cpp
// Iterators: the uniform way to walk through any container.
//
// WHAT YOU LEARN:
//   - begin()/end() give you positions, not values — dereference *it.
//   - ++it moves right; += n jumps; --it / rbegin() go backwards.
//   - Through an iterator you can READ and WRITE the element.
//   - range-for (for x : c) is just a clean wrapper over iterators.
//   - std::distance tells you how far two iterators are apart.
//
// EXPECTED OUTPUT:
//   first element via begin()       : 10
//   jump +2 via operator+=          : 30
//   forward traversal (begin..end)  : 10 20 30 40
//   double via iterator write       : 20 40 60 80
//   reverse traversal (rbegin..rend): 80 60 40 20
//   distance(begin,end) = 4

#include <iostream>
#include <iterator>    // std::distance
#include <vector>

int main() {
    std::vector<int> v{10, 20, 30, 40};

    std::cout << "first element via begin()       : " << *(v.begin())
              << "\n";

    auto it = v.begin();
    it += 2;                                     // jump over two slots
    std::cout << "jump +2 via operator+=          : " << *it << "\n";

    std::cout << "forward traversal (begin..end)  : ";
    for (auto i = v.begin(); i != v.end(); ++i) {
        std::cout << *i << " ";
    }
    std::cout << "\n";

    for (auto i = v.begin(); i != v.end(); ++i) {
        *i = *i * 2;                             // write back through it
    }
    std::cout << "double via iterator write       : ";
    for (int x : v) {                            // range-for = sugar
        std::cout << x << " ";
    }
    std::cout << "\n";

    std::cout << "reverse traversal (rbegin..rend): ";
    for (auto i = v.rbegin(); i != v.rend(); ++i) {
        std::cout << *i << " ";
    }
    std::cout << "\n";

    std::cout << "distance(begin,end) = "
              << std::distance(v.begin(), v.end()) << "\n";
    return 0;
}