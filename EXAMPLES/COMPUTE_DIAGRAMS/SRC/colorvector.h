// colorvector.h
// Dehua Guo
// Aug 2014

#pragma once

#include "cpu_vector.h"
#include "random_vectors.h"

namespace qcd {

struct colorvector :  generic_vector<cpu_buffer<su3_vector, 1> >, qcd::cpu::VectorCpu<double_complex>
{
  colorvector(lattice_desc* desc, bool dirty=false) :
    generic_vector<cpu_buffer<su3_vector, 1> >(desc, desc->sites_on_node, dirty),
    qcd::cpu::VectorCpu<double_complex>(3*desc->sites_on_node, reinterpret_cast<double_complex*>(this->data)) 
  {}

  using  qcd::cpu::VectorCpu<double_complex>::operator = ; 
  // this assignement operator masks the one from generic_vector
  colorvector& operator = (const colorvector& v) { qcd::cpu::VectorCpu<double_complex>::operator = (v); return *this
    ;}
  private:
  colorvector(const colorvector&);
};

void random_vector_uniform(colorvector &vec, random_field& rnd);

}
