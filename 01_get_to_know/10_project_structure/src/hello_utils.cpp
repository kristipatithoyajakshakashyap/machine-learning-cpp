// DEFINITIONS for the declarations made in include/hello_utils.hpp.
// Compiled ONCE into the gtk_utils static library; every executable that
// links it can call these functions.

#include "hello_utils.hpp"

namespace greetings {

std::string cheerful(const std::string& name) {
    return "hi " + name + "!";
}

long square(long x) {
    return x * x;
}

}  // namespace greetings