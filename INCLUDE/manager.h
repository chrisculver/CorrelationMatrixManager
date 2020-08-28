#ifndef MANAGER_H
#define MANAGER_H

#include "OPS/operator.h"
#include "DIAGS/correlator.h"

#include <fstream>
#include <string>
#include <vector>

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


struct Trace_Code_Data
{
  std::vector<std::string> compute_name;
  int idx;///This may not be used anymore

  Trace_Code_Data(std::vector<std::string> name, int idx):compute_name(name), idx(idx){};
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
	///Holds the amount of memory on GPU
	long int gpu_memory;

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
	void load_numerical_results();

	std::vector<Trace> traces_to_compute();
	std::vector<Trace> traces_to_compute(const std::vector<std::string> computed_names);

	void cpu_code_output(std::ofstream &file, std::vector<Trace> need_to_compute);
	void gpu_code_output(std::ofstream &cppfile, std::ofstream &gpufile, std::vector<Trace> need_to_compute);
	void diagram_names_output(std::ofstream &file, std::vector<Trace> need_to_compute);
	void runtime_input_for_cpu(std::ofstream &file, std::vector<Trace> need_to_compute);


	///This function multiplies traces and averages over time slices
	void compute_time_average_correlators();

	void print_correlators();

	///Deletes spdlogger - everything else should delete under RAII
	void shutdown();
};

#endif
