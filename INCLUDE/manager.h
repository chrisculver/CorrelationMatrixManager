#ifndef MANAGER_H
#define MANAGER_H

#include "OPS/operator.h"
#include "DIAGS/correlator.h"

#include <string>

///Contains the ensemble specifications
struct Lattice
{
	///Parameters to specify an ensemble
	int nx,ny,nz,nt,cfg;
	///Default constructor
	Lattice(){nx=0;ny=0;nz=0;nt=0;cfg=0;}
	///Specify all member variables at construction
	Lattice(int x, int y, int z, int t, int c):nx(x),ny(y),nz(z),nt(t),cfg(c){};
};

///Container to specify file names
struct FileNames
{
	///Holds the operator lists
	std::string operator_filename;
	///Holds the diagrams for the given lattice
	std::string diagram_filename;

	FileNames(){operator_filename="";diagram_filename="";};
	FileNames(std::string o_f, std::string d_f):operator_filename(o_f),diagram_filename(d_f){};
};


///Manages the correlation matrix
struct Manager
{
	///Bool to choose how much output there is only the main logger or all loggers
  bool verbose_logging;	
	///Holds the lattice data
	Lattice lat;
	///Holds the filenames
	FileNames files;

	std::vector<Operator> ops;

	std::vector<Correlator> corrs;

	Manager(){};

	///This function parses the input into it's data structures
	void load_input(std::string filename);

  ///Creates the loggers
  void create_logs();

	///This functions loads the operators from a file
	void load_operators();

	///This function performs the wick contractions for all correlation matrix elements
	void wick_contractions();

	///This function loads the diagrams from a file(s). 
//	void load_diagrams();

	///This function multiplies traces and averages over time slices
//	void compute_correlators();

};

#endif
