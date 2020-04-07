// colorvector.cpp
// Dehua Guo
// Aug 2014

#include "colorvector.h"

namespace qcd{

void random_vector_uniform(colorvector &vec, random_field& rnd)
{
#pragma omp parallel for
  for(int iIndex = 0; iIndex < vec.length; iIndex++)
    for(int iColor = 0; iColor < 3; iColor++)
    {
      double real = rnd.data[iIndex].rand();
      double imag = rnd.data[iIndex].rand();
      vec.data[iIndex].c[iColor] = double_complex(real, imag);
    }
}  


}
