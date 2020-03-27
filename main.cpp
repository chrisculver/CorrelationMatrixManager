#include "manager.h"

#include <iostream>
#include <fstream>
#include <string>
#include <map>

using namespace std;

int main(int argc, char **argv)
{
	std::string input_filename = argv[1];
	Manager run;
	try
	{
		run.load_input( input_filename );
		run.load_operators();
	}
	catch(char &ex)
	{
	  if(ex == 'm')
    {
      cout << "Error: Can't open input file\n";
      cout << "       Exiting...\n";
    }
    
    if(ex == 'e')
    {  
      cout << "ERROR: CFG missing from input\n";
      cout << "       Cannot create log files\n";
      cout << "       Exiting...\n";
    }
    ///log error failed loading input
		return 9;
	}




	return 0;
}
