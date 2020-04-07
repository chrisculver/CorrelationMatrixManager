// lapmult.cpp
// Dehua Guo
// Aug 2014

#include "lapmult.h"

namespace qcd{


lapmult::lapmult(su3_field &lnks) :
  lnks(lnks),
  tmp(const_cast<lattice_desc*>(lnks.desc)),
  tmp2(const_cast<lattice_desc*>(lnks.desc))
{
  for (int i=0; i<3; i++) {
    position p(0,0,0,0);
    p[i] = 1;
    shifts[i] = new cached_shift(p, *lnks.desc);
    p[i] = -1;
    shifts[i+3] = new cached_shift(p, *lnks.desc);
  }
}

lapmult::~lapmult()
{
  for (int i=0; i<6; i++)
    delete shifts[i];
}
  
void lapmult::lnk_times_vec(colorvector &src, colorvector &res, int dir, int pm)
{
  for(unsigned int i=0; i<lnks.desc->sites_on_node; i++)
  {
    if(pm == 1)  res.data[i] = lnks.data[i+lnks.desc->sites_on_node*dir] * src.data[i];
    if(pm == -1) res.data[i] = adj(lnks.data[i+lnks.desc->sites_on_node*dir]) * src.data[i];
  }
}


void lapmult::operator()(colorvector &src, colorvector &res)
{
  //clear result 
  res.clear();
  //Sum_(i=1)^3 U_i(x,t) \psi(x+i,t) 
  //shift +x,+y,+z and add to the result
  for(int i=0; i<3; i++){
    shifts[i+3]->shift(src, tmp);
    lnk_times_vec(tmp, tmp2, i, 1);
    res = res + tmp2;
    //vec1_plus_scalar_times_vec2(res, 1.0, js.tmp2, res);  
  }

  //Sum_(i=1)^3 U^{\dag}_i(x-i,t) \psi(x-i,t) 
  //shift -x,-y,-z and add to the result
  for(int i=0; i<3; i++){
    lnk_times_vec(src, tmp, i, -1);
    shifts[i]->shift(tmp, tmp2);
    res = res + tmp2;
    //vec1_plus_scalar_times_vec2(res, 1.0, js.tmp2, res);  
  }
  res = res - 6*src;
  //scalar_times_vector(kappa, res, res); 
}

void lapmult::covderiv(int i, colorvector &src, colorvector &res)
{ //compute U_i(x,t0) psi(x+i,t) -U^{\dag}_i(x-i,t0) \psi(x-i,t)
  res.clear();
  shifts[i+3]->shift(src, tmp);
  lnk_times_vec(tmp, tmp2, i, 1);
  res = res + tmp2;

  lnk_times_vec(src, tmp, i, -1);
  shifts[i]->shift(tmp, tmp2);
  res = res - tmp2;
}

}
