///I submitted some jobs with leading integers for which time slice the diagram was
///computed on but this is redundant info and I want it gone.
///This code takes one of those files and converts it

#include "string_utilities.h"
#include <assert.h>
#include <limits>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <cfloat>
#include <complex>
using namespace std;


///from stackoverflow "How should I do floating point comparison?"
bool nearly_equal(float a, float b,
									float epsilon=FLT_EPSILON, float rel = FLT_MIN)
{
	assert(std::numeric_limits<double>::epsilon() <= epsilon);
	assert(epsilon < 1.0);

	if(a==b) return true;

	auto diff = std::abs(a-b);
	auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
	cout << "     diff=" << diff << "  |  rel=" << rel << "  |  epsilon*norm=" << epsilon*norm << endl;


	return diff < std::max(rel, epsilon*norm);
}


///estimate for ignoring last two digits.
///compares them up to floating point precision.
#define ULP_N 2
bool nearly_equal_abs(complex<double> a, complex<double> b)
{
	double diff = abs(a-b);
	double mag = (abs(a) + abs(b))/2;
	return diff <= (mag*FLT_EPSILON*(1ull << ULP_N));
}


int main(int argc, char **argv)
{
	cout << "Comparing values in diags_cpu.dat and diags_gpu.dat..." << endl;

	std::string cpu_results_name(argv[1]);
	std::string gpu_results_name(argv[2]);

  ifstream cpu_file(cpu_results_name);
	ifstream gpu_file(gpu_results_name);
  std::vector<string> lines;

  string line;
	vector<string> cpu_lines, gpu_lines;
  int n_lines=0;
  while(getline(cpu_file, line))
	{
		if( line[0] != '[' )
		{
			cpu_lines.push_back(line);
			n_lines++;
		}
	}
	while(getline(gpu_file, line))
	{
		if( line[0] != '[' )
			gpu_lines.push_back(line);
	}

	bool diffs = false;
	for(int n=0; n<n_lines; ++n)
	{
		vector<string> c_line = split(cpu_lines[n],' ');
		vector<string> g_line = split(gpu_lines[n],' ');

		for(size_t i=0; i<c_line.size(); ++i)
		{
			vector<string> tmp = split(c_line[i],'e');
			tmp[0] = tmp[0].substr(0,tmp[0].size()-1);
			c_line[i] = tmp[0] + "e" + tmp[1];
			tmp = split(g_line[i],'e');
			tmp[0] = tmp[0].substr(0,tmp[0].size()-1);
			g_line[i] = tmp[0] + "e" + tmp[1];

		}

		for(size_t i=0; i<c_line.size(); i=i+2)
		{
			complex<double> c_val(stod(c_line[i]), stod(c_line[i+1]));
			complex<double> g_val(stod(g_line[i]), stod(c_line[i+1]));

			if( !nearly_equal_abs(c_val, g_val) )
			{
				cout << "found two numbers not nearly equal on line " << n << endl;
				cout << "they are cpu=" << c_val << "   |   gpu=" << g_val << endl;
				diffs = true;
			}
		}
	}


	if(diffs==false)
		cout << "no differences up to floating point precision" << endl;
}
