#ifndef MOM_MATRIX_H
#define MOM_MATRIX_H


/**
 *	\file mom_matrix.h
 *	\brief Header file for mom_matrix functions.
 *	\author Chris Culver
 * 	\date 3/30/2018
 *
 */


#include "laplacean_eigensolver.h"
#include "lapmult.h"
#include "matrix.h"
#include "inject.h"

using namespace qcd;


/**	\brief creats the symmetric "identity" momentum matrix (often operator 1 at gwu)
	* 
	* \param	std::vector<matrix> 			the momentum matrix to write to 
	* \param 	std::vector<int> 					the lattice dimensions(nx,ny,nz,nt)
	* \param 	int 											number of eigenvectors
	* \param 	laplacean_eigensolver 		Solved laplacean system 
	* \param 	int 											x momentum
	*	\param 	int 											y momentum
	*	\param 	int 											z momentum
	*	\param 	int 											node to run on
	*/
void create_matp(std::vector<matrix> &matp, std::vector<int> lat_size, int nvec, laplacean_eigensolver lapeig,
                                int px, int py, int pz, int this_node);



#endif
