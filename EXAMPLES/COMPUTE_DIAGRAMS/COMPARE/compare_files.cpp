#include "string_utilities.h"

#include <string>
#include <iostream>
#include <fstream>
#include <complex>

using namespace std;

bool diagclose(const vector<vector<complex<double>>> a, const vector<vector<complex<double>>> b, const double EPSILON = 1E-8)
{
	auto N = a.size(); 
	for(size_t i=0; i<N; ++i)
		for(size_t j=0; j<N; ++j)
		{
			if(abs(a[i][j] - b[i][j])>EPSILON)
				return false;
		}
	return true;
}

complex<double> largest_diff(const vector<vector<complex<double>>> a, const vector<vector<complex<double>>> b)
{
	auto EPSILON = 1E-12;
	auto N = a.size(); 
	complex<double> res(0,0);
	for(size_t i=0; i<N; ++i)
		for(size_t j=0; j<N; ++j)
		{
			auto diff = a[i][j]-b[i][j];
			if(abs(diff.real()) > res.real())
				res.real( diff.real() );
			if(abs(diff.imag()) > res.imag())
				res.imag( diff.imag() );
		}
	return res;
}


struct Diag
{
	string name;
	vector<vector<complex<double>>> val;
};


vector<Diag> read_diagrams(const string filename, const int NT)
{
	ifstream input(filename);
	string line, name;
	vector<Diag> diags;

	while(getline(input, line))
	{
		vector<vector<complex<double>>> tmp(NT);
		for(size_t i=0; i<NT; ++i)
			tmp[i].resize(NT);

		if(line[0]=='[')
			name=line;
		else
		{
			auto idx=0;
			auto vals = split(line, ' ');
			for(int i=0; i<NT; ++i)
				tmp[idx][i] = complex<double>{stod(vals[2*i]), stod(vals[2*i+1])};
			idx++;
			for(int j=1; j<NT; ++j)
			{
				getline(input, line);
				auto vals = split(line, ' ');
				for(int i=0; i<NT; ++i)
					tmp[idx][i] = complex<double>{stod(vals[2*i]), stod(vals[2*i+1])};
				idx++;
			}
			Diag d;
			d.name = name;
			d.val = tmp;
			diags.push_back(d);
		}
	}
	input.close();
	return diags;
}



int main()
{
	auto andrei_diags = read_diagrams("out", 4);
	auto chris_diags = read_diagrams("diags_4444_100.dat", 4);
	complex<double> biggest_diff(0.,0.);
	double eps = 1E-8;
	for(size_t i=0; i<andrei_diags.size(); ++i)
	{
		if(andrei_diags[i].name!=chris_diags[i].name)
		{
			cout << "Name mismatch at diagram " << i << endl;
			cout << "a name = " << andrei_diags[i].name << endl;
			cout << "c name = " << chris_diags[i].name << endl;
			return 1;
		}

		if(!diagclose(andrei_diags[i].val, chris_diags[i].val, eps))
		{
			cout << "diagram value mismatch abs(delta) > " << eps << " at diagram " << i << endl;
		}

		auto this_diff = largest_diff(andrei_diags[i].val, chris_diags[i].val);
		if(this_diff.real() > biggest_diff.real())
			biggest_diff.real( this_diff.real() );
		if(this_diff.imag() > biggest_diff.imag())
			biggest_diff.imag( this_diff.imag() );
	}

	cout << "biggest difference in real, imag parts are " << biggest_diff << endl;

	return 0;
}
