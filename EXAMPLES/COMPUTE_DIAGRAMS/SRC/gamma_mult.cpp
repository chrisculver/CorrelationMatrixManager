#include "gamma_mult.h"
#include "complex.h"
namespace qcd{
  double_complex g1g5(int i, int j)
  { 
    double_complex tmp(0,0);
    if(i==0&&j==3) tmp.imag = 1; 
    if(i==1&&j==2) tmp.imag = 1;
    if(i==2&&j==1) tmp.imag = 1;
    if(i==3&&j==0) tmp.imag = 1;
    return tmp;
  }
  double_complex g2g5(int i, int j)
  { 
    double_complex tmp(0,0);
    if(i==0&&j==3) tmp.real =  1; 
    if(i==1&&j==2) tmp.real = -1;
    if(i==2&&j==1) tmp.real =  1;
    if(i==3&&j==0) tmp.real = -1;
    return tmp;
  } 
  double_complex g3g5(int i, int j)
  { 
    double_complex tmp(0,0);
    if(i==0&&j==2) tmp.imag =  1; 
    if(i==1&&j==3) tmp.imag = -1;
    if(i==2&&j==0) tmp.imag =  1;
    if(i==3&&j==1) tmp.imag = -1;
    return tmp;
  }
  double_complex g5g5(int i, int j)
  { 
    double_complex tmp(0,0);
    if(i==0&&j==0) tmp.real =  1; 
    if(i==1&&j==1) tmp.real =  1;
    if(i==2&&j==2) tmp.real =  1;
    if(i==3&&j==3) tmp.real =  1;
    return tmp;
  }
  double_complex g1g4g5(int i, int j)
  { 
    double_complex tmp(0,0);
    if(i==0&&j==1) tmp.imag =  -1; 
    if(i==1&&j==0) tmp.imag =  -1;
    if(i==2&&j==3) tmp.imag =  -1;
    if(i==3&&j==2) tmp.imag =  -1;
    return tmp;
  }
  double_complex g2g4g5(int i, int j)
  { 
    double_complex tmp(0,0);
    if(i==0&&j==1) tmp.real =  -1; 
    if(i==1&&j==0) tmp.real =   1;
    if(i==2&&j==3) tmp.real =  -1;
    if(i==3&&j==2) tmp.real =   1;
    return tmp;
  }
  double_complex g3g4g5(int i, int j)
  { 
    double_complex tmp(0,0);
    if(i==0&&j==0) tmp.imag =  -1; 
    if(i==1&&j==1) tmp.imag =   1;
    if(i==2&&j==2) tmp.imag =  -1;
    if(i==3&&j==3) tmp.imag =   1;
    return tmp;
  }
}
