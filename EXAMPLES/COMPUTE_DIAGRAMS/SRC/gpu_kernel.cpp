#include "gpu_kernel.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

/*qcd::matrix cublas_multiply_two(qcd::matrix qcdA, qcd::matrix qcdB, int dim)
{
	std::complex<double> *A = (std::complex<double> *)malloc(sizeof(std::complex<double>)*dim*dim);
	std::complex<double> *B = (std::complex<double> *)malloc(sizeof(std::complex<double>)*dim*dim);

	for(int r=0; r<dim; r++)
	for(int c=0; c<dim; c++)
	{
		A[r*dim+c] = std::complex<double>(qcdA[r*dim+c].real, qcdA[r*dim+c].imag);
		B[r*dim+c] = std::complex<double>(qcdB[r*dim+c].real, qcdB[r*dim+c].imag);
	}


	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

	const unsigned int mem_size = dim*dim*sizeof(std::complex<double>);

	cuDoubleComplex *d_A, *d_B;
	cudaMalloc((void **) &d_A, mem_size);
	cudaMalloc((void **) &d_B, mem_size);

	cudaMemcpy(d_A, A, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, mem_size, cudaMemcpyHostToDevice);

	std::complex<double> alpha(1.,0.);
	std::complex<double> beta(0.,0.);
	cuDoubleComplex *_alpha = reinterpret_cast<cuDoubleComplex*>(&alpha);
	cuDoubleComplex *_beta = reinterpret_cast<cuDoubleComplex*>(&beta);

	int block_size = 32;
	dim3 threads(block_size, block_size);
	dim3 grid(dim/threads.x, dim/threads.y);

	stat = cublasCreate(&handle);

	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_A, dim, d_B, dim, _beta, d_A, dim);

	cudaMemcpy(A, d_A, mem_size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cublasDestroy(handle);

	qcd::matrix res;
	res.resize(dim,dim);

	for(int r=0; r<dim; r++)
	for(int c=0; c<dim; c++)
			res[r*dim+c]=qcd::double_complex(A[r*dim+c].real(),A[r*dim+c].imag());

	free(A);
	free(B);

	return res;
}


qcd::matrix cublas_multiply_four(qcd::matrix qcdA, qcd::matrix qcdB, qcd::matrix qcdC, qcd::matrix qcdD, int dim)
{
	std::complex<double> *A = (std::complex<double> *)malloc(sizeof(std::complex<double>)*dim*dim);
	std::complex<double> *B = (std::complex<double> *)malloc(sizeof(std::complex<double>)*dim*dim);
	std::complex<double> *C = (std::complex<double> *)malloc(sizeof(std::complex<double>)*dim*dim);
	std::complex<double> *D = (std::complex<double> *)malloc(sizeof(std::complex<double>)*dim*dim);

	for(int r=0; r<dim; r++)
	for(int c=0; c<dim; c++)
	{
		A[r*dim+c] = std::complex<double>(qcdA[r*dim+c].real, qcdA[r*dim+c].imag);
		B[r*dim+c] = std::complex<double>(qcdB[r*dim+c].real, qcdB[r*dim+c].imag);
		C[r*dim+c] = std::complex<double>(qcdC[r*dim+c].real, qcdC[r*dim+c].imag);
		D[r*dim+c] = std::complex<double>(qcdD[r*dim+c].real, qcdD[r*dim+c].imag);
	}


	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

	const unsigned int mem_size = dim*dim*sizeof(std::complex<double>);

	cuDoubleComplex *d_A, *d_B, *d_C, *d_D;
	cudaMalloc((void **) &d_A, mem_size);
	cudaMalloc((void **) &d_B, mem_size);
	cudaMalloc((void **) &d_C, mem_size);
	cudaMalloc((void **) &d_D, mem_size);

	cudaMemcpy(d_A, A, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_D, D, mem_size, cudaMemcpyHostToDevice);

	std::complex<double> alpha(1.,0.);
	std::complex<double> beta(0.,0.);
	cuDoubleComplex *_alpha = reinterpret_cast<cuDoubleComplex*>(&alpha);
	cuDoubleComplex *_beta = reinterpret_cast<cuDoubleComplex*>(&beta);

	int block_size = 32;
	dim3 threads(block_size, block_size);
	dim3 grid(dim/threads.x, dim/threads.y);

	stat = cublasCreate(&handle);

	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_A, dim, d_B, dim, _beta, d_A, dim);
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_C, dim, d_D, dim, _beta, d_C, dim);
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_A, dim, d_C, dim, _beta, d_A, dim);

	cudaMemcpy(A, d_A, mem_size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_D);
	cublasDestroy(handle);

	qcd::matrix res;
	res.resize(dim,dim);

	for(int r=0; r<dim; r++)
	for(int c=0; c<dim; c++)
			res[r*dim+c]=qcd::double_complex(A[r*dim+c].real(),A[r*dim+c].imag());
	
	free(A);
	free(B);
	free(C);
	free(D);

	return res;
}

*/
void cublas_batch_multiply_two(std::complex<double> *A, std::complex<double> *B, int batch, int dim)
{
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	cudaStream_t stream;

	cudaStreamCreate(&stream);
	stat = cublasCreate(&handle);

	cublasSetStream(handle, stream);

	const unsigned int mem_size = dim*dim*batch*sizeof(std::complex<double>);

	cuDoubleComplex *d_A, *d_B, *d_C;
	cudaMalloc((void **) &d_A, mem_size);
	cudaMalloc((void **) &d_B, mem_size);
	cudaMalloc((void **) &d_C, mem_size);

	cudaMemcpy(d_A, A, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, mem_size, cudaMemcpyHostToDevice);

	std::complex<double> alpha(1.,0.);
	std::complex<double> beta(0.,0.);
	cuDoubleComplex *_alpha = reinterpret_cast<cuDoubleComplex*>(&alpha);
	cuDoubleComplex *_beta = reinterpret_cast<cuDoubleComplex*>(&beta);

	int block_size = 32;
	dim3 threads(block_size, block_size);
	dim3 grid(dim/threads.x, dim/threads.y);

	cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_A, dim, dim*dim, d_B, dim, dim*dim, _beta, d_C, dim, dim*dim, batch);

	cudaFree(d_B);
	cudaFree(d_A);
	cudaMemcpy(A, d_C, mem_size, cudaMemcpyDeviceToHost);
	cudaFree(d_C);
	cublasDestroy(handle);
	cudaStreamSynchronize(0);
	cudaStreamDestroy(stream);
}






void cublas_batch_multiply_four(std::complex<double> *A, std::complex<double> *B, std::complex<double> *C, std::complex<double> *D, int batch, int dim)
{
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	cudaStream_t stream;

	cudaStreamCreate(&stream);
	stat = cublasCreate(&handle);

	cublasSetStream(handle, stream);

	const unsigned int mem_size = dim*dim*batch*sizeof(std::complex<double>);

	cuDoubleComplex *d_A, *d_B, *d_C, *d_D, *d_E;
	cudaMalloc((void **) &d_A, mem_size);
	cudaMalloc((void **) &d_B, mem_size);
	cudaMalloc((void **) &d_E, mem_size);

	cudaMemcpy(d_A, A, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, mem_size, cudaMemcpyHostToDevice);

	std::complex<double> alpha(1.,0.);
	std::complex<double> beta(0.,0.);
	cuDoubleComplex *_alpha = reinterpret_cast<cuDoubleComplex*>(&alpha);
	cuDoubleComplex *_beta = reinterpret_cast<cuDoubleComplex*>(&beta);

	int block_size = 32;
	dim3 threads(block_size, block_size);
	dim3 grid(dim/threads.x, dim/threads.y);

	cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_A, dim, dim*dim, d_B, dim, dim*dim, _beta, d_E, dim, dim*dim, batch);
	cudaFree(d_B);
	cudaMalloc((void **) &d_C, mem_size);
	cudaMemcpy(d_C, C, mem_size, cudaMemcpyHostToDevice);
	cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_E, dim, dim*dim, d_C, dim, dim*dim, _beta, d_A, dim, dim*dim, batch);
	cudaFree(d_C);
	cudaMalloc((void **) &d_D, mem_size);
	cudaMemcpy(d_D, D, mem_size, cudaMemcpyHostToDevice);
	cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_A, dim, dim*dim, d_D, dim, dim*dim, _beta, d_E, dim, dim*dim, batch);
	cudaFree(d_D);
	cudaFree(d_A);
	cudaMemcpy(A, d_E, mem_size, cudaMemcpyDeviceToHost);
	cudaFree(d_E);
	cublasDestroy(handle);
	cudaStreamSynchronize(0);
	cudaStreamDestroy(stream);

}







void cublas_batch_multiply_six(std::complex<double> *A,  std::complex<double> *B, std::complex<double> *C, std::complex<double> *D, std::complex<double> *E, std::complex<double> *F, int batch, int dim)
{
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	cudaStream_t stream;

	cudaStreamCreate(&stream);
	stat = cublasCreate(&handle);
	cublasSetStream(handle, stream);

	const unsigned int mem_size = dim*dim*batch*sizeof(std::complex<double>);

	cuDoubleComplex *d_A, *d_B, *d_C, *d_D, *d_E, *d_F, *d_G;
	cudaMalloc((void **) &d_A, mem_size);
	cudaMalloc((void **) &d_B, mem_size);
	cudaMalloc((void **) &d_G, mem_size);

	cudaMemcpy(d_A, A, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, mem_size, cudaMemcpyHostToDevice);

	std::complex<double> alpha(1.,0.);
	std::complex<double> beta(0.,0.);
	cuDoubleComplex *_alpha = reinterpret_cast<cuDoubleComplex*>(&alpha);
	cuDoubleComplex *_beta = reinterpret_cast<cuDoubleComplex*>(&beta);

	int block_size = 32;
	dim3 threads(block_size, block_size);
	dim3 grid(dim/threads.x, dim/threads.y);

	cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_A, dim, dim*dim, d_B, dim, dim*dim, _beta, d_G, dim, dim*dim, batch);
	cudaFree(d_B);
	cudaMalloc((void **) &d_C, mem_size);
	cudaMemcpy(d_C, C, mem_size, cudaMemcpyHostToDevice);
	cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_G, dim, dim*dim, d_C, dim, dim*dim, _beta, d_A, dim, dim*dim, batch);

	cudaFree(d_C);
	cudaMalloc((void **) &d_D, mem_size);
	cudaMemcpy(d_D, D, mem_size, cudaMemcpyHostToDevice);
	cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_A, dim, dim*dim, d_D, dim, dim*dim, _beta, d_G, dim, dim*dim, batch);

	cudaFree(d_D);
	cudaMalloc((void **) &d_E, mem_size);
	cudaMemcpy(d_E, E, mem_size, cudaMemcpyHostToDevice);
	cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_G, dim, dim*dim, d_E, dim, dim*dim, _beta, d_A, dim, dim*dim, batch);

	cudaFree(d_E);
	cudaMalloc((void **) &d_F, mem_size);
	cudaMemcpy(d_F, F, mem_size, cudaMemcpyHostToDevice);
	cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, _alpha, d_A, dim, dim*dim, d_F, dim, dim*dim, _beta, d_G, dim, dim*dim, batch);

	cudaFree(d_F);
	cudaFree(d_A);
	
	cudaMemcpy(A, d_G, mem_size, cudaMemcpyDeviceToHost);

	cudaFree(d_G);
	cublasDestroy(handle);
	cudaStreamSynchronize(0);
	cudaStreamDestroy(stream);
}
