#ifndef FILE_UTILITIES_H
#define FILE_UTILITIES_H

#include <string>
#include <map>
#include <vector>
#include <complex>

bool file_exists(std::string filename);

std::map<std::string, std::vector<std::vector<std::complex<double>>> > parse_diagram_file(std::string filename, int NT);

std::string cpp_prefix();
std::string cpp_postfix();

#endif
