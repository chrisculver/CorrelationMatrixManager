///I submitted some jobs with leading integers for which time slice the diagram was
///computed on but this is redundant info and I want it gone.  
///This code takes one of those files and converts it

#include "string_utilities.h"

#include <fstream>
#include <string>

using namespace std;


string time(string full)
{
	if(full=="f")
		return "1";
	else if(full=="i")
		return "0";
}

int main(int argc, char **argv)
{
	std::string original_file = "diagram_names.txt";
  ifstream diag_file(original_file);
  std::vector<string> lines;
  
  ofstream outfile;
  outfile.open("in");

  outfile << "latname CFG/c44_b5.3_k0.158_100" << endl;
  outfile << "nx 4" << endl << "ny 4" << endl << "nz 4" << endl << "nt 4" << endl;
  outfile << "nvec 10" << endl;
 
  string line; 
  int n_lines=0;
  while(getline(diag_file, line))
  {
    n_lines++;
    lines.push_back(line);
  }

  outfile << "diagrams:length " << n_lines << endl;

  int n=0;
  for(const auto& l : lines)
  {
    outfile << "diagrams:" << n << "  ";
    n++;
    auto split_line = split(l, '|');
    for(auto s : split_line)
    {
      auto term = split(s,' ');
		//	if(term[0]!="[")
		//	{	
		//		outfile << gamma(term[0]) << " ";
		//		outfile << term[2] << " " << term[3] << " " << term[4];
		//		outfile << term[5] << " ";
		//	}
		//	else
		//	{
				outfile << term[1] << " ";
				outfile << term[3] << " " << term[4] << " " << term[5] << " ";
				outfile << time(term[6]) << " ";
		//	}
			
			/*
      for(auto t : term)
      {
        if(t=="\\Gamma_5")
          outfile << "5";
				else if(t=="\\Gamma_x")
					outfile << "1";
        else if(t=="t_f")
          outfile << "1";
        else if(t=="t")
          outfile << "0";
        else if( (t=="[") || (t=="]") ){}
				else if( t=="\\delta_{ii}") {}
        else
          outfile << t;

        if(t!=term.back())
          outfile << " ";
        else
          outfile << "   ";
      }*/
    }
    outfile << endl;
  }

  outfile.close();


}
