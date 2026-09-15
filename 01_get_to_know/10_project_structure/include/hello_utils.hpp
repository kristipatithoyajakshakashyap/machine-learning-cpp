#ifndef HELLO_UTILS_HPP
#define HELLO_UTILS_HPP

// DECLARATIONS only. The definitions live in src/hello_utils.cpp.
// Any .cpp that needs these functions just #includes this header.

#include <string>

namespace greetings {

// Say hi, warmly.
std::string cheerful(const std::string& name);

// Square an integer.
long square(long x);

}  // namespace greetings

#endif  // HELLO_UTILS_HPP