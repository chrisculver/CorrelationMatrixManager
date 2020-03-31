#include "manager.h"

#include "UTILS/string_utilities.h"

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
	  if(ex == 'f')
    {
      cout << "CRITICAL ERROR: Can't open input file\n";
      cout << "								 Exiting...\n";
    }
    
    if(ex == 'c')
    {  
      cout << "CRITICAL ERROR: CFG missing from input\n";
      cout << "								 Cannot create log files\n";
      cout << "								 Exiting...\n";
    }
		if(ex == 'i')
		{
			cout << "CRITICAL ERROR: Extra input variable found\n";
			cout << "								 See main_"+cfg_to_string(run.lat.cfg)+".log";
			cout << "					       Exiting...\n";
		}
		if(ex == 'o')
		{
			cout << "CRITICAL ERROR: Couldn't find operator file\n";
			cout << "                Exiting...\n";
		}
		return 9;
	}

	run.wick_contractions();


	run.shutdown();

	return 0;
}
