#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include <complex>
#include <vector>
//#include "matrix.h"

//qcd::matrix cublas_multiply_two(qcd::matrix qcdA, qcd::matrix qcdB, int dim);
//qcd::matrix cublas_multiply_four(qcd::matrix qcdA, qcd::matrix qcdB, qcd::matrix qcdC, qcd::matrix qcdD, int dim);

void cublas_batch_multiply_two(std::complex<double> *A, std::complex<double> *B, int batch, int dim);
void cublas_batch_multiply_four(std::complex<double> *A, std::complex<double> *B, std::complex<double> *C, std::complex<double> *D, int batch, int dim);
void cublas_batch_multiply_six(std::complex<double> *A, std::complex<double> *B,	std::complex<double> *C, std::complex<double> *D, std::complex<double> *E, std::complex<double> *F, int batch, int dim);

#endif
