#include "manager.h"

#include "UTILS/string_utilities.h"

#include <fstream> 
#include <map>

using namespace std;


/*!
 *	Constructor for the manager class which takes the main input file.
 *
 *	The input file contains a data member on each line with a key and a value.
 *	We read all keys and put them into the appropriate variables
 *	Once a key is used it is deleted, then at the end if any keys remain 
 *	We give an error and exit.  Either there is an extra harmless variable, 
 *	or a key was mistypes - which would cause major issues
 */
void Manager::load_input(string input_filename)
{
	ifstream input(input_filename);
	map<string, string> name_value;

	string line;
	while( getline(input, line) )
	{
		auto data = split(line,' ');
		name_value[ data[0] ] = data[1];
	}
	
	lat = Lattice( stoi(name_value["nx"]), 
								 stoi(name_value["ny"]), 
								 stoi(name_value["nz"]), 
								 stoi(name_value["nt"]), 
								 stoi(name_value["cfg"]) );
	name_value.erase("nx");
	name_value.erase("ny");
	name_value.erase("nz");
	name_value.erase("nt");
	name_value.erase("cfg");

	///log output as it's read in

	if(name_value.size() > 0)
	{
		///add error log output

		for(auto &e : name_value)
		{
			///add error log output
			///"Key=" << e.first << " | Value=" << e.second << endl;
		}
		throw 'i';
	}

}


void Manager::load_operators()
{

}
