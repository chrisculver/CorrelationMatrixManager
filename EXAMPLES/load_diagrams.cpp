#include <fstream>
#include <string>
#include <map>
#include <complex>
#include <iostream>
#include <vector>

using namespace std;
using Saved_Diagrams = map<string, map<string,complex<double>>>;

template<typename Out> void split(const string &s, char delim, Out result)
{
  stringstream ss(s);
  string item;
  while(getline(ss, item, delim))
  {
    *(result++) = item;
  }
}

vector<string> split(const string&s, char delim)
{
  vector<string> elems;
  split(s, delim, back_inserter(elems));
  return elems;
}

int main(int argc, char **argv)
{
  ifstream file(argv[1]);
  string line;
  string current_diagram;
  Saved_Diagrams res;
  while(getline(file, line))
  {
    if(line[0]=='[')
      current_diagram=line;
    else
    {
      auto columns = split(line, ' ');
      res[current_diagram][columns[0]+" "+columns[1]]=std::complex<double>{stod(columns[2]), stod(columns[3])};
    }
  }

  for(const auto &c : res)
  {
    cout << c.first << endl;
    auto data = c.second;
    for(const auto &d : data)
      cout << d.first << "   " << d.second << endl;
  }

  return 0;
}
