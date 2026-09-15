// 07_stl_core/02_vector_and_list.cpp
// vector vs list: where each one is fastest.
//
// WHAT YOU LEARN:
//   - vector< >: elements sit side by side; appending at the BACK is cheap
//     (amortized O(1)). Inserting at the front must shift everything.
//   - list< >:   elements are linked nodes; inserting anywhere is cheap
//     (O(1)) but you pay for node allocator bookkeeping.
//   - insert/erase take an iterator position.
//
// EXPECTED OUTPUT:
//   vector after 3 push_backs  : 10 20 30
//   vector after insert 5 front: 5 10 20 30
//   vector after erase middle  : 5 20 30
//   vector size = 3
//   list  after front+back     : 0 1 2 3
//   list  size = 4

#include <iostream>
#include <list>
#include <vector>

template <typename Container>
void print(const char* label, const Container& c) {
    std::cout << label << ": ";
    for (const auto& x : c) {
        std::cout << x << " ";
    }
    std::cout << "\n";
}

int main() {
    std::vector<int> v;
    for (int i = 1; i <= 3; ++i) {
        v.push_back(i * 10);                 // cheap at the back
    }
    print("vector after 3 push_backs  ", v);

    v.insert(v.begin(), 5);                  // expensive: shifts all
    print("vector after insert 5 front", v);

    v.erase(v.begin() + 1);                  // remove the '10'
    print("vector after erase middle  ", v);
    std::cout << "vector size = " << v.size() << "\n";

    std::list<int> l;
    l.push_back(2);
    l.push_front(1);                         // cheap on a list
    l.push_front(0);
    l.push_back(3);
    print("list  after front+back     ", l);
    std::cout << "list  size = " << l.size() << "\n";
    return 0;
}