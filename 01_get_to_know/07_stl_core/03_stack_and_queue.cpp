// 07_stl_core/03_stack_and_queue.cpp
// Stack and queue: two strict orderings that organize work.
//
// WHAT YOU LEARN:
//   - stack: LAST IN, FIRST OUT — push(top), pop(top), top().
//     The most recent task is handled first.
//   - queue: FIRST IN, FIRST OUT — push(back), pop(front), front().
//     Tasks are handled in the order they arrived.
//   - deque: the double-ended queue behind both; you push at both ends.
//
// EXPECTED OUTPUT:
//   stack  (LIFO) pops in this order : 3 2 1
//   queue  (FIFO) pops in this order : 1 2 3
//   deque  front add + back add      : [9 1 2 3 5]

#include <deque>
#include <iostream>
#include <queue>
#include <stack>

int main() {
    // --- stack: last pushed is first out ---
    std::stack<int> st;
    st.push(1);
    st.push(2);
    st.push(3);
    std::cout << "stack  (LIFO) pops in this order : ";
    while (!st.empty()) {
        std::cout << st.top() << " ";
        st.pop();
    }
    std::cout << "\n";

    // --- queue: first pushed is first out ---
    std::queue<int> q;
    q.push(1);
    q.push(2);
    q.push(3);
    std::cout << "queue  (FIFO) pops in this order : ";
    while (!q.empty()) {
        std::cout << q.front() << " ";
        q.pop();
    }
    std::cout << "\n";

    // --- deque: push back + push front ---
    std::deque<int> d;
    d.push_back(1);
    d.push_back(2);
    d.push_back(3);
    d.push_front(9);
    d.push_back(5);
    std::cout << "deque  front add + back add      : [";
    for (size_t i = 0; i < d.size(); ++i) {
        std::cout << d[i] << (i + 1 < d.size() ? " " : "");
    }
    std::cout << "]\n";
    return 0;
}