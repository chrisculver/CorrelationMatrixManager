// smeared_matrix_object.h
// A. Alexandru
// Aug 2014

#pragma once

#include <vector>
#include <string>
#include "complex.h"

namespace qcd {

struct smeared_matrix_object
{
  int nt, nvec;
  std::vector<double_complex> vals;
  smeared_matrix_object(int nt_, int nvec_) : nt(nt_), nvec(nvec_), vals(nt*nvec*4*nt*nvec*4) {}

  double_complex& operator()(int t, int v, int s, int tp, int vp, int sp)
  {
    return vals[t+nt*(v+nvec*(s+4*(tp+nt*(vp+nvec*sp))))];
  }
};

void write(const std::string& mat_file, const smeared_matrix_object& m, const std::string& latname);

void read(const std::string& mat_file, smeared_matrix_object& m, std::string& latname);

}
