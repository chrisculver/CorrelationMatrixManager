// lapmult.h
// Dehua Guo
// Aug 2014

#pragma once

#include "colorvector.h"
#include "qcd_fields.h"
#include "comm/comm_shift.h"
#include "matmult.h"


namespace qcd {

struct lapmult : matmult<colorvector>
{
  su3_field &lnks;
  colorvector tmp, tmp2;
  cached_shift *shifts[6];

  lapmult(su3_field &lnks);
  ~lapmult();
  void lnk_times_vec(colorvector &src, colorvector &res, int dir, int pm);
  void operator()(colorvector &src, colorvector &res);
  void covderiv(int dir, colorvector &src, colorvector &res);
}; // end of lapmult

}
