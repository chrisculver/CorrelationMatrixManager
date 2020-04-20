#ifndef STRING_UTILITIES_H
#define STRING_UTILITIES_H

#include <string>
#include <vector> 

template<typename Out> void split(const std::string &s, char delim, Out result);
std::vector<std::string> split(const std::string &s, char delim);

std::string convert_term(std::string term, const std::vector<std::string> input_mom,
																					 std::vector<std::string> corr_mom);


#endif
