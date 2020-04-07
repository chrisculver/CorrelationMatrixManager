#include "mom_matrix.h"

#include<stdio.h>
/**
 *	\file mom_matrix.cpp
 *	\brief Code to compute mom matrices.
 *	\author Chris Culver
 * 	\date 3/30/2018
 *
 */

using namespace qcd;

void create_matp(std::vector<matrix> &matp, std::vector<int> lat_size, int nvec, laplacean_eigensolver lapeig,
                                int px, int py, int pz, int this_node)
{
  int nx=lat_size[0]; int ny=lat_size[1]; int nz=lat_size[2]; int nt=lat_size[3];

  for(int t=0; t<nt; t++)
  {
    for(int c=0; c<nvec; c++)
    {
      colorvector tmp(&lapeig.desc);
      tmp = *lapeig.eigsys(t).evec[c];

#pragma omp parallel for default(none) shared(lapeig, tmp, nx, ny, nz, nt, px, py, pz, this_node), schedule(static)
      for(int i=0; i<lapeig.desc.sites_on_node; i++)
      {
        position p = lapeig.desc.get_position(i,this_node);
        double_complex phase = exp(double_complex(0,1)*2*M_PI*(p[0]*px/((double)nx) + p[1]*py/((double)ny) + p[2]*pz/((double)nz)));
        tmp.data[i] *= phase;
      }

      for(int r=0; r<nvec; r++)
        matp[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r],tmp);
    }
  }

}








