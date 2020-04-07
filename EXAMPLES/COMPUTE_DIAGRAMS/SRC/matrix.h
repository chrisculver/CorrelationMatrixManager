// matrix.h
// A. Alexandru
// Sep 2014


#pragma once

#include "complex.h"
#include "Eigen/Dense"

namespace qcd
{

struct matrix : Eigen::Matrix<double_complex, Eigen::Dynamic, Eigen::Dynamic>
{
  matrix(int nr=1, int nc=1) : Eigen::Matrix<double_complex, Eigen::Dynamic, Eigen::Dynamic>(nr, nc) {}

  double_complex& operator[](int n) { return *(this->data()+n); }

  using Eigen::Matrix<double_complex, Eigen::Dynamic, Eigen::Dynamic>::operator= ;
};

}
