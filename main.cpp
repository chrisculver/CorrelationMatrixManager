#include "manager.h"

#include "UTILS/string_utilities.h"

#include <iostream>
#include <fstream>
#include <string>
#include <map>

using namespace std;

int main(int argc, char **argv)
{
	std::string input_filename = argv[1];
	Manager run;
	try
	{
		run.load_input( input_filename );
		run.load_operators();
	}
	catch(char &ex)
	{
	  if(ex == 'f')
    {
      cout << "CRITICAL ERROR: Can't open input file\n";
      cout << "								 Exiting...\n";
    }

    if(ex == 'c')
    {
      cout << "CRITICAL ERROR: CFG missing from input\n";
      cout << "								 Cannot create log files\n";
      cout << "								 Exiting...\n";
    }
		if(ex == 'i')
		{
			cout << "CRITICAL ERROR: Extra input variable found\n";
			cout << "								 See main_"+cfg_to_string(run.lat.cfg)+".log";
			cout << "					       Exiting...\n";
		}
		if(ex == 'o')
		{
			cout << "CRITICAL ERROR: Couldn't find operator file\n";
			cout << "                Exiting...\n";
		}
		return 9;
	}

	run.wick_contractions();

	try
	{
		///TODO: Currently all diagrams in one file, goes through the list
		//sequentially, load's alot of data into memory, and searches a large map.
		///Could create some clever system to look in different files corresponding
		///to the length of the diagram and load all those traces, then next size, etc.
		run.load_numerical_results();
	}
	catch(vector<Trace<QuarkLine>> need_to_compute)
	{
		ofstream cpu_code("define_diagrams.cpp");
		run.cpu_code_output(cpu_code, need_to_compute);
		cpu_code.close();

		ofstream gpu_compute_cppfile("define_diagrams_gpu.cpp");
		ofstream gpu_compute_cudafile("gpu_kernel.cpp");
		run.gpu_code_output(gpu_compute_cppfile, gpu_compute_cudafile, need_to_compute);
		gpu_compute_cppfile.close();
		gpu_compute_cudafile.close();


		ofstream diag_names("diagram_names.txt");
		run.diagram_names_output(diag_names, need_to_compute);
		diag_names.close();

		ofstream run_in("run_c44.in");
		run.runtime_input_for_cpu(run_in, need_to_compute);
		run_in.close();

		return 1;
	}

	run.compute_time_average_correlators();

	run.print_correlators();

	run.shutdown();

	return 0;
}
