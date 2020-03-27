#ifndef STRING_UTILITIES_H
#define STRING_UTILITIES_H

#include <string>
#include <vector>


///splits a string using delim, from stackoverflow
template<typename Out> void split(const std::string &s, char delim, Out result);
std::vector<std::string> split(const std::string &s, char delim);

std::string cfg_to_string(int cfg);

#endif
