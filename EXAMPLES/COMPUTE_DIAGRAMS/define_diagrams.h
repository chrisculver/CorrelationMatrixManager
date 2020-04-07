#ifndef COMPUTE_DIAGRAMS_H
#define COMPUTE_DIAGRAMS_H

#include <vector>
#include <complex>
#include <Eigen/Dense>

void define_diagrams(std::vector< std::vector< std::vector< std::complex<double> > > > &diag,
                     std::vector<Eigen::MatrixXcd> qf, std::vector<Eigen::MatrixXcd> qb,
                     std::vector<Eigen::MatrixXcd> qti, std::vector<Eigen::MatrixXcd> qtf,
                     int t, int dt);

#endif
