#ifndef MANAGER_H
#define MANAGER_H

#include <string>

struct Lattice
{
	int nx,ny,nz,nt,cfg;

	Lattice(){nx=0;ny=0;nz=0;nt=0;cfg=0;}
	Lattice(int x, int y, int z, int t, int c):nx(x),ny(y),nz(z),nt(t),cfg(c){};
};


struct Manager
{
  bool verbose_logging;	
	Lattice lat;
//	File_Manager fm;

	Manager(){};

	///This function parses the input into it's data structures
	void load_input(std::string filename);

  ///Creates the loggers
  void create_logs();

	///This functions loads the operators from a file
	void load_operators();

	///This function performs the wick contractions for all correlation matrix elements
//	void wick_contractions();

	///This function loads the diagrams from a file(s). 
//	void load_diagrams();

	///This function multiplies traces and averages over time slices
//	void compute_correlators();

};

#endif
