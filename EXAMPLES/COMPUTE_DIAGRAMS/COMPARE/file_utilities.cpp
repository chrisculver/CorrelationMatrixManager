#include "file_utilities.h"
#include "string_utilities.h"

#include <sstream>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <algorithm>

using namespace std;

vector<Diagram> read_diagrams(const std::string filename, int NT)
{
  std::ifstream input(filename);
  std::string line;

  std::vector<Diagram> diags;

  std::string name;
	while(getline(input, line))
	{
    std::vector<std::vector<std::complex<double>>> tmp(NT);
		for(int t=0; t<NT; t++)
			tmp[t].resize(NT);
    
		if(line[0]=='[')
			name=line;

		else
		{
			auto idx=0;
			auto vals = split(line, ' ');
			for(int i=0; i<NT; ++i)
				tmp[idx][i] = std::complex<double>{stod(vals[2*i]), stod(vals[2*i+1])};
			idx++;
			for(int j=1; j<NT; ++j)
			{
				getline(input, line);
				auto vals = split(line, ' ');
				for(int i=0; i<NT; ++i)
					tmp[idx][i] = std::complex<double>{stod(vals[2*i]), stod(vals[2*i+1])};
				idx++;
			}
			diags.push_back(Diagram(name,term_text_to_trace(name),tmp));
		}
	}

	input.close();

	return diags;
}

Trace term_text_to_trace(string term_text)
{
	vector<string> terms = split(term_text, '|');
	vector<Quark_Line> qlines;
	for(size_t i=0; i<terms.size(); ++i)
	{
		vector<string> t = split(terms[i],' ');
	  ///note it fills with an empty string after the split from |	
    qlines.push_back(Quark_Line(t[1],t[2]+" "+t[3]+" "+t[4],t[5]));
	}
	return Trace(qlines);
}

void Diagram::printvals(std::ostream &out) const
{
  for(const auto& dt : value)
  {
    for(const auto& t : dt)
    {
      out << scientific << setprecision(10) << t.real() << " " << t.imag() << " ";
    }
    out << endl;
  }
}


void Diagram::fix_pion_pion()
{
  for(auto& d : value) 
    for(auto& t : d)
      t*=-1.;
}

void Diagram::fix_pipi_pipi()
{
  cout << "Error.  NOTHING IMPLEMENTED";
}

void Diagram::fix_pipipi_pipipi()
{
  cout << "Error.  NOTHING IMPLEMENTED";
}
