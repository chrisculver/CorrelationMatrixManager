#include "file_utilities.h"

#include <string>
#include <iostream>
#include <complex>

using namespace std;

bool diagclose(const vector<vector<complex<double>>> a, const vector<vector<complex<double>>> b)
{
  auto EPSILON=1E-8;
  auto N = a.size();
  for(int i=0; i<N; ++i)
    for(int j=0; j<N; ++j)
      if(abs(a[i][j]-b[i][j])>EPSILON)
			{
        cout << i << "," << j;
				return false;  
			}
	return true;
}


int main()
{
  cout << "Starting..." << endl;
  auto andrei_out="out";
  auto andreidiags=read_diagrams(andrei_out,4);
//  cout << "d0=" << andreidiags[0].name << endl;
//  andreidiags[0].printvals();
  auto chris_out="diags.dat";
  auto chrisdiags=read_diagrams(chris_out,4);
//  cout << "d0=" << chrisdiags[0].name << endl;
//  chrisdiags[0].printvals();


  int tmp=andreidiags.size(); 
 /* 
  for(size_t i=0; i<andreidiags.size(); ++i)
  {
    for(size_t j=0; j<chrisdiags.size(); ++j)
    {
    */
  for(size_t i=0; i<tmp; ++i)
  {
//    for(size_t j=0; j<chrisdiags.size(); ++j)
//    {
		int j=i;
      if(andreidiags[i].name == chrisdiags[j].name)
      if(!diagclose(andreidiags[i].value,chrisdiags[j].value))
      {
        cout << "MISMATCH at i= " << i << ",j=" << j << endl;
        cout << "andrei = " << andreidiags[i].name << endl;
        andreidiags[i].printvals();
        cout << "chris = " << chrisdiags[j].name << endl;
        chrisdiags[j].printvals();
        break;
      }
//    }
  }
  
  return 0;
}
