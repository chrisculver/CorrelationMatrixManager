#include "UTILS/file_utilities.h"
#include "UTILS/string_utilities.h"
#include "manager.h"

#include <fstream>

using namespace std;
using Saved_Traces = map<string, vector<vector<complex<double>>> >;

bool file_exists(string filename)
{
  ifstream file(filename.c_str());
  return file.good();
}

Saved_Traces parse_diagram_file(std::string filename, int NT)
{
	ifstream input(filename);
  string line;
  vector<string> all_names;
  vector<vector<vector<complex<double>>>> all_values;
  while(getline(input, line))
  {
		vector<vector<complex<double>>> tmp_val(NT);
    for(int t=0; t<NT; ++t)
      tmp_val[t].resize(NT);

    if(line[0]=='[')
      all_names.push_back(line); 
    else
    {
      auto idx=0;
      auto vals=split(line, ' ');
      for(int i=0; i<NT; ++i)
        tmp_val[idx][i] = std::complex<double>{stod(vals[2*i]), stod(vals[2*i+1])};
      idx++;
      for(int j=1; j<NT; ++j)
      {
        getline(input, line);
        auto vals = split(line, ' ');
        for(int i=0; i<NT; ++i)
          tmp_val[idx][i] = std::complex<double>{stod(vals[2*i]), stod(vals[2*i+1])};
        idx++;
      }
      all_values.push_back(tmp_val);
    }
  } 
  input.close();

  Saved_Traces computed_traces;
  for(size_t i=0; i<all_names.size(); ++i)
    computed_traces[all_names[i]] = all_values[i];

  return computed_traces; 

}

string cpp_prefix()
{
  string text;

  text += "#include \"define_diagrams.h\"\n";
	text += "typedef Eigen::MatrixXcd mat;\n";
	text += "using namespace std;\n";
	text += "void define_diagrams(vector< vector< vector < complex<double> > > > &diag, const vector<mat> qf, const vector<mat> qb,";
  text += "const vector<mat> qti, const vector<mat> qtf, int t, int dt)\n";
	text += "{\n";

  return text;
}

string cpp_postfix()
{
  return "\n}";
}
