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
		auto split_endpoint = split(l, ']');
    auto split_line = split(split_endpoint[0], '|');
    for(auto s : split_line)
    {
      auto term = split(s,' ');

			if(term.size()==8)	
			{	
				outfile << term[1] << " ";
				outfile << term[3] << " " << term[4] << " " << term[5] << " ";
				outfile << time(term[6]) << " ";
			}
			else if(term.size()==9)	
			{	
				if(term[1]=="1" && term[2]=="5")
					outfile << "11 ";
				else if(term[1]=="2" && term[2]=="5")
					outfile << "12 ";
				else if(term[1]=="3" && term[2]=="5")
					outfile << "13 ";
				
				outfile << term[4] << " " << term[5] << " " << term[6] << " ";
				outfile << time(term[7]) << " ";
			}
			
    }
    outfile << endl;
  }

  outfile.close();


}
