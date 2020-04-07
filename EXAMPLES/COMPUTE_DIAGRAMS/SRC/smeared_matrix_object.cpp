// smeared_matrix_object.cpp
// A. Alexandru
// Aug 2014

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "smeared_matrix_object.h"

namespace qcd {


void write(const std::string& mat_file, const smeared_matrix_object& m, const std::string& latname)
{
  FILE* f = fopen(mat_file.c_str(), "wb");
  char cfgname[100]; strncpy(cfgname, latname.c_str(), 100);
  fwrite(cfgname, 1, 100, f);
  uint32_t _nt = m.nt; uint32_t _nvec = m.nvec;
  fwrite(&_nt, 4, 1, f); fwrite(&_nvec, 4, 1, f); 
  fwrite(&m.vals[0], sizeof(double_complex), m.vals.size(), f);
  fclose(f);
}

void read(const std::string& mat_file, smeared_matrix_object& m, std::string& latname)
{
  FILE* f = fopen(mat_file.c_str(), "rb");
  char cfgname[100]; 
  fread(cfgname, 1, 100, f); cfgname[99] = 0;
  latname = std::string(cfgname);
  uint32_t nt, nvec;
  fread(&nt, 4, 1, f); fread(&nvec, 4, 1, f);
  m.nt = nt; m.nvec = nvec; m.vals.resize(nt*nvec*4*nt*nvec*4); 
  fread(&m.vals[0], sizeof(double_complex), m.vals.size(), f);
  fclose(f);
}

}
