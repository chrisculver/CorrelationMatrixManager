#include "UTILS/string_utilities.h"

#include <sstream>

using namespace std;

vector<string> split(const string&s, char delim)
{
  vector<string> elems;
  split(s, delim, back_inserter(elems));
  return elems;
}

template<typename Out> void split(const string &s, char delim, Out result)
{
  stringstream ss(s);
  string item;
  while(getline(ss, item, delim))
  {
    *(result++) = item;
  }
}
