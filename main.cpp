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
	catch(exception &ex)
	{
		///log error failed loading input
		return 9;
	}









	return 0;
}
