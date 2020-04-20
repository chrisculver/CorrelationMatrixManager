#ifndef FILE_UTILITIES_H
#define FILE_UTILITIES_H

#include "diagrams.h"

#include <string>
#include <vector>

#include <complex>
#include <iostream>

struct Diagram
{
  std::string name;
  Trace trace;
  std::vector<std::vector<std::complex<double>>> value;

  Diagram(std::string nname, Trace ntrace,std::vector<std::vector<std::complex<double>>> nvalue):name(nname),trace(ntrace),value(nvalue){};
  void printvals(std::ostream &out=std::cout) const;

  void time_reverse_two_names();
  void fix_pion_pion();
  void fix_pipi_pipi();
  void fix_pipipi_pipipi();
};

Trace term_text_to_trace(std::string term_text);
std::vector<Diagram> read_diagrams(const std::string filename, int NT);

#endif
