#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include <complex>
#include <vector>

void cublas_batch_multiply_1_0(std::complex<double> *res, std::vector<std::complex<double>*> qf, 
																													std::vector<std::complex<double>*> qb,
																													std::vector<std::complex<double>*> qti,
																													std::vector<std::complex<double>*> qtf,
																													long int dim
																													);
void cublas_batch_multiply_2_0(std::complex<double> *res, std::vector<std::complex<double>*> qf, 
																													std::vector<std::complex<double>*> qb,
																													std::vector<std::complex<double>*> qti,
																													std::vector<std::complex<double>*> qtf,
																													long int dim
																													);
void cublas_batch_multiply_2_1(std::complex<double> *res, std::vector<std::complex<double>*> qf, 
																													std::vector<std::complex<double>*> qb,
																													std::vector<std::complex<double>*> qti,
																													std::vector<std::complex<double>*> qtf,
																													long int dim
																													);
void cublas_batch_multiply_3_0(std::complex<double> *res, std::vector<std::complex<double>*> qf, 
																													std::vector<std::complex<double>*> qb,
																													std::vector<std::complex<double>*> qti,
																													std::vector<std::complex<double>*> qtf,
																													long int dim
																													);
void cublas_batch_multiply_3_1(std::complex<double> *res, std::vector<std::complex<double>*> qf, 
																													std::vector<std::complex<double>*> qb,
																													std::vector<std::complex<double>*> qti,
																													std::vector<std::complex<double>*> qtf,
																													long int dim
																													);
void cublas_batch_multiply_4_0(std::complex<double> *res, std::vector<std::complex<double>*> qf, 
																													std::vector<std::complex<double>*> qb,
																													std::vector<std::complex<double>*> qti,
																													std::vector<std::complex<double>*> qtf,
																													long int dim
																													);
void cublas_batch_multiply_4_1(std::complex<double> *res, std::vector<std::complex<double>*> qf, 
																													std::vector<std::complex<double>*> qb,
																													std::vector<std::complex<double>*> qti,
																													std::vector<std::complex<double>*> qtf,
																													long int dim
																													);
void cublas_batch_multiply_5_0(std::complex<double> *res, std::vector<std::complex<double>*> qf, 
																													std::vector<std::complex<double>*> qb,
																													std::vector<std::complex<double>*> qti,
																													std::vector<std::complex<double>*> qtf,
																													long int dim
																													);
void cublas_batch_multiply_5_1(std::complex<double> *res, std::vector<std::complex<double>*> qf, 
																													std::vector<std::complex<double>*> qb,
																													std::vector<std::complex<double>*> qti,
																													std::vector<std::complex<double>*> qtf,
																													long int dim
																													);
void cublas_batch_multiply_6_0(std::complex<double> *res, std::vector<std::complex<double>*> qf, 
																													std::vector<std::complex<double>*> qb,
																													std::vector<std::complex<double>*> qti,
																													std::vector<std::complex<double>*> qtf,
																													long int dim
																													);
void cublas_batch_multiply_6_1(std::complex<double> *res, std::vector<std::complex<double>*> qf, 
																													std::vector<std::complex<double>*> qb,
																													std::vector<std::complex<double>*> qti,
																													std::vector<std::complex<double>*> qtf,
																													long int dim
																													);

#endif
