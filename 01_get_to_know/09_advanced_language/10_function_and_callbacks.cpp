// 09_advanced_language/10_function_and_callbacks.cpp
// Passing behaviour around: std::function, lambdas, functors, function
// pointers, std::bind, and a tiny event-handler registry.
//
// WHAT YOU LEARN:
//   - A callback is "code handed to someone else to run later". C++ has
//     three callable shapes: plain function pointers, functors (objects
//     with operator()), and lambdas (compiler-generated functors).
//   - std::function<R(Args...)> is a type-erased wrapper that can hold ANY
//     of those shapes, so one vector can store mixed callbacks.
//   - std::bind glues arguments to a callable; a lambda does the same job
//     more readably and usually inlines better. Prefer the lambda.
//   - Event handlers are just a map from event name to a list of callbacks.
//   - Cost note: a template parameter (auto/F) lets the compiler inline the
//     call; std::function adds an indirect call and may heap-allocate large
//     captures. Use templates in hot loops, std::function at API boundaries.
//
// EXPECTED OUTPUT:
//   apply(free function)  -> 30
//   apply(functor)        -> 200
//   apply(lambda)         -> 24
//   function pointer add  -> 7
//   bind add(_1, 10)      -> 15
//   lambda add(x, 10)     -> 15
//   stored callbacks: 4 -> 13 9 6 -7
//   event "save" fired: logger got save
//   event "save" fired: counter = 1
//   event "quit" fired: logger got quit
//   event "load" has no handlers
//   templated sum = 55, std::function sum = 55

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// --- three callable shapes -------------------------------------------------
int times_three(int x) { return x * 3; }              // free function

struct Scale {                                        // functor
    int factor;
    int operator()(int x) const { return x * factor; }
};

// std::function accepts any of them: type erasure.
int apply(const std::function<int(int)>& f, int value) { return f(value); }

// --- function pointer and std::bind ---------------------------------------
int add(int a, int b) { return a + b; }

// --- event handler mini example -------------------------------------------
class EventBus {
public:
    using Handler = std::function<void(const std::string&)>;

    void subscribe(const std::string& event, Handler h) {
        handlers_[event].push_back(std::move(h));
    }

    void fire(const std::string& event) const {
        auto it = handlers_.find(event);
        if (it == handlers_.end()) {
            std::cout << "event \"" << event << "\" has no handlers\n";
            return;
        }
        for (const Handler& h : it->second) h(event);
    }

private:
    std::map<std::string, std::vector<Handler>> handlers_;
};

// --- cost note: template parameter vs std::function ------------------------
template <typename F>
int sum_with_template(const std::vector<int>& v, F f) {   // inlinable
    int total = 0;
    for (int x : v) total += f(x);
    return total;
}

int sum_with_function(const std::vector<int>& v,
                      const std::function<int(int)>& f) { // indirect call
    int total = 0;
    for (int x : v) total += f(x);
    return total;
}

int main() {
    std::cout << "apply(free function)  -> " << apply(times_three, 10) << '\n';
    std::cout << "apply(functor)        -> " << apply(Scale{20}, 10) << '\n';
    std::cout << "apply(lambda)         -> "
              << apply([](int x) { return x + 14; }, 10) << '\n';

    int (*fp)(int, int) = &add;                       // raw function pointer
    std::cout << "function pointer add  -> " << fp(3, 4) << '\n';

    using namespace std::placeholders;
    auto add_ten_bind = std::bind(add, _1, 10);       // classic
    auto add_ten_lambda = [](int x) { return add(x, 10); };  // preferred
    std::cout << "bind add(_1, 10)      -> " << add_ten_bind(5) << '\n';
    std::cout << "lambda add(x, 10)     -> " << add_ten_lambda(5) << '\n';

    // A vector of mixed callbacks, all under one std::function type.
    std::vector<std::function<int(int)>> callbacks;
    callbacks.push_back(add_ten_bind);
    callbacks.push_back(times_three);
    callbacks.push_back(Scale{2});
    callbacks.push_back([](int x) { return x * x - 16; });
    std::cout << "stored callbacks: " << callbacks.size() << " ->";
    for (const auto& cb : callbacks) std::cout << ' ' << cb(3);
    std::cout << '\n';

    // Event handlers: subscribe two lambdas (one captures by reference).
    EventBus bus;
    int save_count = 0;
    bus.subscribe("save", [](const std::string& e) {
        std::cout << "event \"" << e << "\" fired: logger got " << e << '\n';
    });
    bus.subscribe("save", [&save_count](const std::string& e) {
        ++save_count;
        std::cout << "event \"" << e << "\" fired: counter = " << save_count << '\n';
    });
    bus.subscribe("quit", [](const std::string& e) {
        std::cout << "event \"" << e << "\" fired: logger got " << e << '\n';
    });
    bus.fire("save");
    bus.fire("quit");
    bus.fire("load");

    const std::vector<int> nums{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto identity = [](int x) { return x; };
    std::cout << "templated sum = " << sum_with_template(nums, identity)
              << ", std::function sum = " << sum_with_function(nums, identity)
              << '\n';
    return 0;
}
