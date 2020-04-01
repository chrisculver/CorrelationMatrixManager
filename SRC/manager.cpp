#include "manager.h"

#include "OPS/operator.h"
#include "OPS/elemental_op.h"
#include "OPS/meson.h"
#include "UTILS/string_utilities.h"
#include "UTILS/file_utilities.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/fmt/ostr.h"

#include <iostream>
#include <fstream> 
#include <map>
#include <string>

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
	///Make sure the input file exists, if not error
	if( !file_exists(input_filename) )
  {
    throw 'f';
  }
  
  
  ifstream input(input_filename);
	map<string, string> name_value;

	string line;
	while( getline(input, line) )
	{
		auto data = split(line,' ');
		name_value[ data[0] ] = data[1];
	}

	///We label logs files by cfg, so make sure this input variable exists.
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
	auto main_logger = spdlog::get("main");
	main_logger->info("Loaded lattice data\nnx={} | ny={} | nz={} | nt={} | cfg={}\n",
										 lat.nx, lat.ny, lat.nz, lat.nt, cfg_to_string(lat.cfg));	
	
	files = FileNames(name_value["operator_filename"], name_value["diagram_filename"]);
	name_value.erase("operator_filename");
	name_value.erase("diagram_filename");	

	main_logger->info("Loaded filenames\noperator_filename = {} | diagram_filename = {}\n",
									  files.operator_filename, files.diagram_filename );

	if(!verbose_logging)
		main_logger->info("Loaded all input data\n");

	if(name_value.size() > 0)
	{
		///TODO add error log output
		
		for(auto &e : name_value)
		{
			///TODO  error log output
			///"Key=" << e.first << " | Value=" << e.second << endl;
		}
    ///i for input error
		throw 'i';
	}

	main_logger->flush();
}

void Manager::create_logs()
{
  ///We always create one log file to show what input was read in
  ///and to track major points in control flow.  
  auto main_logger = spdlog::basic_logger_mt("main", "logs/main_"+cfg_to_string(lat.cfg)+".log");
	main_logger->set_level(spdlog::level::info);
  main_logger->info("Program started up.");
  ///Just doing one extra type of output - debug/verbose
  auto wick_logger = spdlog::basic_logger_mt("wick", "logs/wick_"+cfg_to_string(lat.cfg)+".log"); 
	auto op_logger = spdlog::basic_logger_mt("op", "logs/op_"+cfg_to_string(lat.cfg)+".log");
}

void Manager::load_operators()
{
  ///TODO Have better error messages when things break!
	///Several parts can seg fault if the operator's have a typo in them.
	auto main_logger = spdlog::get("main");
  main_logger->info("Begin loading operators");
	main_logger->info("Warning: Bad error handling ahead\n");
	
	auto op_logger = spdlog::get("op");

	if( !file_exists(files.operator_filename) )
		throw 'o';
	
	ifstream op_file(files.operator_filename);
	string line;
	while(getline(op_file,line))
	{
    vector<ElementalOp> elems;
		op_logger->debug("Read line = {}",line);

		auto op_sum = split(line, '+');
    int coef;
    for(const auto term:op_sum)
    {
      auto meson_text = split(term, '|');
			coef = stoi(meson_text[0]); 
			meson_text.erase(meson_text.begin(), meson_text.begin()+1);
      vector<Meson> ms;
      for(const auto meson:meson_text)
      {
        vector<string> data = split(meson,',');
				ms.push_back(Meson(data[0][0],data[1],data[2],data[3],data[4][0]));
      }
      elems.push_back(ElementalOp(coef, ms));
    }
    ops.push_back(Operator(coef,elems));
	} 
  op_file.close();
	

	main_logger->info("Finished loading the following operators");
	for(size_t i=0; i<ops.size(); ++i)
		main_logger->info("Operator {} = {}", i, ops[i]);

	for(const auto &c : ops)
		for(const auto &a : ops)
			corrs.push_back( Correlator(adjoint(a), c) );

	op_logger->flush();
	main_logger->flush();
}


void Manager::wick_contractions()
{
	for(auto &c: corrs)
		c.wick_contract();

	for(size_t i=0; i<ops.size(); ++i)
	for(size_t j=0; j<ops.size(); ++j)
	{
		auto wick_logger = spdlog::get("wick");
		std::string diag_names;
		for(const auto &d: corrs[i*ops.size() + j].diags)
			diag_names+=d.name()+"\n";	
		wick_logger->info("Resulting diags for c_{}{} = \n{}", i, j, diag_names);
	}
}


void Manager::shutdown()
{
	spdlog::shutdown();
}
