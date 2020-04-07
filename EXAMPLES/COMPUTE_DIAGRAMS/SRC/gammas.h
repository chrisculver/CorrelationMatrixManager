#ifndef GAMMAS_H
#define GAMMAS_H

#include "gamma_mult.h"
#include "matrix.h"

using namespace qcd;

double_complex gammaid(int r, int c, bool dag=false);
double_complex gamma5(int r, int c, bool dag=false);
double_complex gammai(int i, int r, int c, bool dag=false);
double_complex gammaigamma5(int i, int r, int c, bool dag=false);

#endif
