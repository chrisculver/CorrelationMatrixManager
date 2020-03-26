#include "manager.h"

#include "UTILS/string_utilities.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

#include <fstream> 
#include <map>
#include <sstream>
#include <iomanip>

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

  if( name_value.count("cfg")==0 )
    throw 'c';


  verbose_logging = stoi(name_value["verbose_logging"]);
  name_value.erase("verbose_logging");

  ///Construct lattice data container
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

  ///Set up logging system
  create_logs();


	///log output as it's read in

	if(name_value.size() > 0)
	{
		///add error log output

		for(auto &e : name_value)
		{
			///add error log output
			///"Key=" << e.first << " | Value=" << e.second << endl;
		}
    ///i for input error
		throw 'i';
	}

}

void Manager::create_logs()
{
  stringstream scfg;
  scfg << std::setfill('0') << std::setw(3) << lat.cfg;

  ///We always create one log file to show what input was read in
  ///and to track major points in control flow.  
  auto main_logger = spdlog::basic_logger_mt("main", "logs/main_"+scfg.str()+".log");
  
  main_logger->info("Program started up.");
  ///Just doing one extra type of output - debug/verbose
  if(verbose_logging)
  {
    auto wick_logger = spdlog::basic_logger_mt("wick", "logs/wick_"+scfg.str()+".log"); 
  }
}

void Manager::load_operators()
{
  auto main_logger = spdlog::get("main");
  main_logger->info("Begin loading operators");
}
