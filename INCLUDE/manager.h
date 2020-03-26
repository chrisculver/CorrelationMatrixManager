#ifndef MANAGER_H
#define MANAGER_H

#include <string>


class Manager
{
	Lattice lat;
	File_Manager fm;

	Manager(std::string input_filename);

	///This functions loads the operators from a file
	void load_operators();

	///This function performs the wick contractions for all correlation matrix elements
	void wick_contractions();

	///This function loads the diagrams from a file(s). 
	void load_diagrams();

	///This function multiplies traces and averages over time slices
	void compute_correlators();

};

#endif
