#include "string_utilities.h"

#include <sstream>

using namespace std;

///Split is used to split a string with the possibility of splitting
///with an arbitrary delimiter.  This was gathered from the internet.  
vector<string> split(const string &s, char delim)
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


string convert_term(string term, const vector<string> input_mom, vector<string> corr_mom)
{
      ////BEGIN CONVERT TERM FUNCTION
      ///conver the pi to actual momenta
	  	auto tmp = split(term, ' ');
	  	string new_term = "[";
	  	for(auto& t: tmp)
	  	{
	  		for(size_t p=0; p<input_mom.size(); ++p)
	  		{
	  			if(t==input_mom[p])
	  				t = corr_mom[p];
	  		}
	  		new_term = new_term + t;
        if(t!="]")
          new_term = new_term + " ";
	  		}
			///? END CONVERT TERM FUNCTION, RETURN NEW_TERM;
  return new_term;
}
