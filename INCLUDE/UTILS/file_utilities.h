#ifndef FILE_UTILITIES_H
#define FILE_UTILITIES_H

#include <string>
#include <map>
#include <vector>
#include <string>
#include <complex>

std::vector<std::string> load_diagram_filenames(std::string file, std::string cfg);

bool file_exists(std::string filename);

void parse_diagram_file(std::map< std::string, std::vector<std::vector<std::complex<double>>>> &computed,
                        std::string filename, int NT);

std::string cpp_prefix();
std::string cpp_postfix();
std::string gpu_code_cpp_prefix();
std::string gpu_code_cpp_postfix();
std::string gpu_code_cuda_prefix();
std::string gpu_code_function_prefix(std::string l, std::string s);
std::string gpu_code_function_postfix();


#endif
