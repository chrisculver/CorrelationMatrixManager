#ifndef IO_H
#define IO_H


/**
 *	\file io.h
 *	\brief Header file for input/output functions.
 *	\author Chris Culver
 * 	\date 3/30/2018
 *
 */


//#include "include_all.h"

#include "smeared_matrix_object.h"
#include "laplacean_eigensolver.h"

#include "options.h"

#include <string>
#include <vector>

/** \brief Reads mtilde
	*	
 	*	\param 	&smeared_matrix_object 			Object to write result
 	*	\param 	std::string 								Name of mtilde file
 	*	\param 	&std::string									Name of link file
	*
	*/
void read_mtilde(qcd::smeared_matrix_object &m, std::string mname, std::string &latname);


/**	\brief Read laplacean eigensystem
	*	
	*	\param &laplacean_eigensolver				Object to write result
	* \param smeared_matrix_object				 
	* \param bool													Check eigensystem
	*	\param &int													Number of eigenvectors
	*	\param std::string												Name of laplacean system file
	*/
void read_lap(qcd::laplacean_eigensolver &lap, qcd::smeared_matrix_object m, bool checkeig, int &nvec, std::string lname);

/**	\brief Read and nhyp smear links
	*	
	*	\param &su3_field 				Object to write result
	*	\param std::string			Name of link file
	*/
void prep_links(qcd::su3_field &l, std::string latname);


/**	\brief Translate command link options for use in code
	*
	*	\param int 														Number of cmd line args
	*	\param int 														The arguments
	*	\param &std::vector<int>		 					Write lattice size here
	* \param &std::vector<std::string>			Write file names here
	*	\param &std::vector<textline>				Write momentum values here
	*	\param &bool													Write check eigensystem here
	*	\param &int														Write number of eigenvectors here
	*	\param &int 													Write cfg here
	*/
void get_opts(int argc, char **argv, std::vector<int> &size, std::vector<std::string> &name, std::vector<qcd::textline> &mom1,  bool &checkeig, int &nvec, int &cfg);


#endif
