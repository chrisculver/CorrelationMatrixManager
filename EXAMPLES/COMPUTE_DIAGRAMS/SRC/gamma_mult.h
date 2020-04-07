#pragma once

#include "matrix.h"
#include "complex.h"
namespace qcd{
  double_complex g1g5(int i, int j);
  double_complex g2g5(int i, int j);
  double_complex g3g5(int i, int j);
  double_complex g5g5(int i, int j);
  double_complex g1g4g5(int i, int j);
  double_complex g2g4g5(int i, int j);
  double_complex g3g4g5(int i, int j);
 
  static struct { 
    matrix g1() { matrix res(4,4); res(0,3) = res(1,2) = double_complex(0,-1); res(2,1) = res(3,0) = double_complex(0,1); return res;}
    matrix g2() { matrix res(4,4); res(0,3) = res(3,0) = 1; res(1,2) = res(2,1) = -1; return res;}
    matrix g3() { matrix res(4,4); res(0,2) = res(3,1) = double_complex(0,-1); res(1,3) = res(2,0) = double_complex(0,1); return res;}
    matrix g4() { matrix res(4,4); res(0,2) = res(1,3) = res(2,0) = res(3,1) = -1; return res;}
    matrix g5() { matrix res(4,4); res =  g1()*g2()*g3()*g4(); return res; }
// Cg5=ig2 g4
    matrix Cg5() { matrix res(4,4); res(0,1) = res(2,3) = double_complex(0,-1); res(1,0) = res(3,2) = double_complex(0,1); return res;}
    matrix C() { matrix res(4,4); res(0,1) = res(3,2) = double_complex(0,-1); res(1,0) = res(2,3) = double_complex(0,1); return res;}
// Cg5= g2g4
//  matrix Cg5() { matrix res(4,4); res(0,1) = res(2,3) = double_complex(-1,0); res(1,0) = res(3,2) = double_complex(1,0); return res;}
  } gamma_matrix;
}
