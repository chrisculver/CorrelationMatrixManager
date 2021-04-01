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

	gpu_memory = stoi(name_value["gpu_memory"]);
	name_value.erase("gpu_memory");

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


	///dt and t to compute for

	if(name_value["dts"]=="ALL")
	{
		for(size_t i=0; i<lat.nt; ++i)
			dts.push_back(i);
	}
	else if(name_value["dts"]=="HALF")
	{
		for(size_t i=0; i<=lat.nt/2; ++i)
			dts.push_back(i);
	}
	else
	{
		vector<string> dts_string = split(name_value["dts"],',');
		for(auto &s : dts_string)
			dts.push_back( stoi(s) );
	}

	if(name_value["ts"]=="ALL")
	{
		for(size_t i=0; i<lat.nt; ++i)
			ts.push_back(i);
	}
	else if(name_value["ts"]=="HALF")
	{
		for(size_t i=0; i<lat.nt/2; ++i)
			ts.push_back(i);
	}
	else
	{
		vector<string> ts_string = split(name_value["ts"],',');
		for(auto &s : ts_string)
			ts.push_back( stoi(s) );
	}

	name_value.erase("dts");
	name_value.erase("ts");

  ///Set up logging system
  create_logs();
	auto main_logger = spdlog::get("main");
	main_logger->info("Loaded lattice data\nnx={} | ny={} | nz={} | nt={} | cfg={}\n",
										 lat.nx, lat.ny, lat.nz, lat.nt, cfg_to_string(lat.cfg));

	//files = FileNames(name_value["operator_filename"], "diags_"+to_string(lat.nx)+to_string(lat.ny)
	//			+to_string(lat.nz)+to_string(lat.nt)+"_"+cfg_to_string(lat.cfg)+".dat");

	files = FileNames(name_value["operator_filename"],
					load_diagram_filenames(name_value["diagram_files"],cfg_to_string(lat.cfg)));
	name_value.erase("operator_filename");
	name_value.erase("diagram_files");

	main_logger->info("Loaded operator_filename = {} \n",
									  files.operator_filename );

	main_logger->info("Using these diagram files");
	for(const auto &d : files.diagram_files)
		main_logger->info("{}",d);

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
	wick_logger->set_level(spdlog::level::info);
	//spdlog::flush_on(spdlog::level::debug);
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
    for(const auto term:op_sum)
    {
      auto meson_text = split(term, '|');
			int coef = stoi(meson_text[0]);
			meson_text.erase(meson_text.begin(), meson_text.begin()+1);
      vector<Meson> ms;
      for(const auto meson:meson_text)
      {
        vector<string> data = split(meson,',');
				ms.push_back(Meson(data[0][0],data[1],data[2],data[3],data[4][0]));
      }
      elems.push_back(ElementalOp(coef, ms));
    }
    ops.push_back(Operator(elems));
	}
  op_file.close();


	main_logger->info("Finished loading the following operators");
	for(size_t i=0; i<ops.size(); ++i)
		main_logger->info("Operator {} = {}", i, ops[i]);

	main_logger->info("The adjoint operators are");
	for(size_t i=0; i<ops.size(); ++i)
		main_logger->info("Adjoint operator {} = {}", i, adjoint(ops[i]));

//	for(const auto &c : ops)
//		for(const auto &a : ops)
//			corrs.push_back( Correlator(adjoint(a), c) );
	for(size_t i=0; i<ops.size(); ++i)
	for(size_t j=0; j<ops.size(); ++j)
	{
		if(j<=i)
			corrs.push_back( Correlator<QuarkLine>(adjoint(ops[i]), ops[j], ts, dts) );
		else
			corrs.push_back( Correlator<QuarkLine>() );
	}

	op_logger->flush();
	main_logger->flush();
}


void Manager::wick_contractions()
{
//	for(auto &c: corrs)
//		c.wick_contract();

	bool load = false;
	string wick_file = "logs/" + files.operator_filename + ".wick";
	if(file_exists(wick_file))
		load=true;

	auto wick_logger = spdlog::get("wick");
	auto main_logger = spdlog::get("main");

	for(size_t i=0; i<ops.size(); ++i)
	for(size_t j=0; j<ops.size(); ++j)
	{

		main_logger->info("Starting contractions for c_{}.{}",i,j);
	//	cout << "effectively reduces the container size by the number of elementswick contraction for c_ij = " << i << " " << j << endl;
		if(load==false)
		{
			corrs[i*ops.size() +j].wick_contract();

			std::string diag_names;
			for(const auto &d: corrs[i*ops.size() + j].diags)
			{
				diag_names+=d.name();
				///compare addresses because its a range based for
				if(&d != &(corrs[i*ops.size()+j].diags).back())
					diag_names+="+";

			}
			wick_logger->info("Resulting diags for c_{}.{}={}", i, j, diag_names);
			main_logger->info("Finished contraction for c_{}.{}",i,j);
		}
		else
		{
			if(j<=i)
				corrs[i*ops.size() + j].load_wick_contractions(wick_file, i, j);
		}
	//	cout << endl;
	}
/*
	for(size_t i=0; i<ops.size(); ++i)
	for(size_t j=0; j<ops.size(); ++j)
	{
		if(i<=j)
		{
			std::cout << "diagrams for i=" << i << " j=" << j << std::endl;
			auto tmp = corrs[i*ops.size()+j];
			for(auto d : tmp.diags)
			{
				std::cout << d.name() << std::endl;
			}
			std::cout << std::endl;
		}
	}
*/

/*	for(size_t i=0; i<ops.size(); ++i)
	for(size_t j=0; j<ops.size(); ++j)
	{
		auto wick_logger = spdlog::get("wick");
		std::string diag_names;
		for(const auto &d: corrs[i*ops.size() + j].diags)
			diag_names+=d.name()+"\n";
		wick_logger->info("Resulting diags for c_{}{} = \n{}", i, j, diag_names);
	}
*/
}


void Manager::load_numerical_results()
{
	auto main_logger = spdlog::get("main");

	for(size_t i=0; i<files.diagram_files.size(); ++i)
	{
		if(!file_exists(files.diagram_files[i]))
		{
			main_logger->info("Diagram file {} not found", files.diagram_files[i]);
			files.diagram_files.erase(files.diagram_files.begin()+i);
		}
	}

	if( files.diagram_files.size()==0 )
	{
		main_logger->info("No diagram files found");
		throw traces_to_compute();
	}

	map<string, map<string,complex<double>>> computed;
	for(const auto d_file : files.diagram_files)
	{
			auto new_diags = parse_diagram_file(d_file);
			for(const auto &d : new_diags)
			{
				if(computed.count(d.first)==0)
					computed[d.first]=d.second;
				else
					computed[d.first].insert(d.second.begin(), d.second.end());
			}
	}

	//computed = parse_diagram_file(files.diagram_filename);

	///   ./compute a1.in none.data
	///   ./compute a1.in daigs.dat
	///   for all diags.dat
	///       NDIAGS   files_used


	for(auto &c: corrs)
	{
		try
		{
			c.load_numerical_results(computed);
		}
		catch(char missing)
		{
			main_logger->info("load_numerical_results threw {}", missing);
			if(missing == 't')
			{
				std::cout << "WARNING: OUTPUTTING ALL TRACES - NO LOOKUP";
				main_logger->info("throwing t, missing diagram routine isn't implemented");
				/// This is because all diagram names match, only certain diagrams will
				/// be missing a certain set of dt, t combos.
				throw traces_to_compute();
			}
			vector<string> computed_names;
			for(const auto &c1 : computed)
				computed_names.push_back(c1.first);
			throw traces_to_compute(computed_names);
		}
	}
}


vector<Trace<QuarkLine>> Manager::traces_to_compute()
{

	auto main_logger = spdlog::get("main");
  main_logger->info("Organizing traces to compute with none computed");
	vector<Trace<QuarkLine>> res;
	for(auto &c: corrs)
	for(auto &d: c.diags)
	for(auto &t: d.traces)
	{
		Trace<QuarkLine> r = t;
		bool found=false;
		for(size_t i=0; i<r.qls.size(); ++i)
		{
			if( find(res.begin(), res.end(), r) != res.end() )
				found=true;
			rotate( r.qls.begin(), r.qls.begin()+1, r.qls.end() );
		}
		if(!found)
			res.push_back(t);
	}
	main_logger->info("Done organizing");
	return res;
}

vector<Trace<QuarkLine>> Manager::traces_to_compute(const vector<string> computed_names)
{
	auto main_logger = spdlog::get("main");
	main_logger->info("Organizing traces to compute with some computed");
	vector<Trace<QuarkLine>> res;
	for(auto &c: corrs)
	for(auto &d: c.diags)
	for(auto &t: d.traces)
	{
		Trace<QuarkLine> r = t;
		bool found=false;
		for(size_t i=0; i<r.qls.size(); ++i)
		{
			if( find(res.begin(), res.end(), r) != res.end() )
				found=true;
			if( find(computed_names.begin(), computed_names.end(), r.name()) != computed_names.end() )
				found=true;
			rotate( r.qls.begin(), r.qls.begin()+1, r.qls.end() );
		}
		if(!found)
			res.push_back(t);
	}
	main_logger->info("Done organizing");
	return res;
}



void Manager::cpu_code_output(ofstream &file, vector<Trace<QuarkLine>> need_to_compute)
{
	auto main_logger = spdlog::get("main");
	main_logger->info("Generating cpp file to compute traces");
  file << cpp_prefix();

  ///The important bit, depends on diagrams!
  vector<string> unique_mom, unique_disp, unique_gamma;
  for(const auto& t : need_to_compute)
    for(const auto& q : t.qls)
    {
      if(find(unique_mom.begin(), unique_mom.end(), q.mom) == unique_mom.end())
        unique_mom.push_back(q.mom);
      if(find(unique_disp.begin(), unique_disp.end(), q.displacement) == unique_disp.end())
        unique_disp.push_back(q.displacement);
      if(find(unique_gamma.begin(), unique_gamma.end(), q.gamma) == unique_gamma.end())
        unique_gamma.push_back(q.gamma);
    }

  int max_size=0;
  for(auto t : need_to_compute)
    if(t.qls.size() > max_size)
      max_size=t.qls.size();
  ///collect diagrams of certain lengths 1,2,3,4,5,6,etc and save their strings
  vector<int> idx_track(max_size,0);///tracks idx for each length
  vector<int> res_idx;///for the result to pull the correct trace
  vector<vector<Trace_Code_Data>> traces_by_size(max_size);///first vector contains traces of a different size, second vector all traces

	///Reorganize the list of traces into traces_by_size.
  for(size_t i=0; i<need_to_compute.size(); ++i)
  {
    auto t_size = need_to_compute[i].qls.size()-1;
    traces_by_size[t_size].push_back(
      Trace_Code_Data(need_to_compute[i].compute_name(unique_mom, unique_disp, unique_gamma), idx_track[t_size])
                                    );
    res_idx.push_back(idx_track[t_size]);
    idx_track[t_size]++;
  }

	///Results of the calculation go here
  for(size_t l=0; l<traces_by_size.size(); ++l)
    if(traces_by_size[l].size()>0)
      file << "std::vector<mat> res" << l+1 << "(" << traces_by_size[l].size()
          << ");" << endl;

  for(size_t l=0; l<traces_by_size.size(); ++l)
  {
    auto trs = traces_by_size[l];
    if(l<2) ///Don't bother looking up precomputed values, don't exist if length = 1 or 2
    {
      for(size_t d=0; d<trs.size(); ++d)
      {
        file << "res" << l+1 << "[" << d << "]=";
        //for(auto q : trs[d].compute_name)
        for(size_t i=0; i<trs[d].compute_name.size(); ++i)
        {
          file << trs[d].compute_name[i];
          if(i!=trs[d].compute_name.size()-1)
            file << "*";
          else
            file << ";";
        }
        file << endl;
      }
    }///end l<2
    else
    {
      for(size_t d=0; d<trs.size(); ++d)
      {
        file << "res" << l+1 << "[" << d << "]=";

        vector<string> computation = trs[d].compute_name;
        bool max_cse=false;  	///if only two mults are left, we can stop
        ///Do a lookup to see if it's possible to substitute something that has
        ///already been computed.  I want to check for the longest list of
        ///mults down to mults of length 2.
        ///This is not a perfect routine as it may substitute for example , A*B*C*D -> B*C even if A*B and C*D are computed.
        for(size_t chk=l-1; chk>=1; chk--)///length of substitution to look for 1 is diagrams of lenth 2
        {
          for(size_t i=0; i<traces_by_size[chk].size(); ++i)///try them all
          {
            auto lookfor = traces_by_size[chk][i].compute_name;
            auto search_res = search(computation.begin(), computation.end(),
                                     lookfor.begin(), lookfor.end());
            if( search_res != computation.end() )
            {
              ///If you found a substitution, put it before the elements,
              ///refind the original mults, and remove them.
              computation.insert( search_res, "res"+to_string(lookfor.size())+"["+to_string(i)+"]" );
              search_res = search(computation.begin(), computation.end(),
                                       lookfor.begin(), lookfor.end());
              computation.erase( search_res, search_res + lookfor.size() );

              trs[d].compute_name = computation;
              if(trs[d].compute_name.size()==2)
                max_cse=true;
            }///end if
            if(max_cse)
              break;
          }///end i loop
          if(max_cse)
            break;
        }///end chk loop

        ///now just output the new expression
        for(size_t i=0; i<trs[d].compute_name.size(); ++i)
        {
          file << trs[d].compute_name[i];
          if(i!=trs[d].compute_name.size()-1)
            file << "*";
          else
            file << ";";
        }
        file << endl;

      }
    }
  }

	///Now every elem of res holds a matrix that needs to be traced over.
  for(size_t d=0; d<need_to_compute.size(); ++d)
    for(size_t l=0; l<traces_by_size.size(); ++l)
      if(need_to_compute[d].qls.size()==(l+1))
        file << "diag[" << d << "][dt][t] = res" << l+1 << "[" << res_idx[d] << "].trace();" << endl;

  file << cpp_postfix();
	main_logger->info("CPU code output");
}

void Manager::diagram_names_output(ofstream &file, vector<Trace<QuarkLine>> need_to_compute)
{
	for(const auto &t: need_to_compute)
		file << t.name() << endl;
}


void Manager::runtime_input_for_cpu(ofstream &file, vector<Trace<QuarkLine>> need_to_compute)
{
  vector<string> unique_mom, unique_disp, unique_gamma;
  for(const auto& t : need_to_compute)
    for(const auto& q : t.qls)
    {
      if(std::find(unique_mom.begin(), unique_mom.end(), q.mom) == unique_mom.end())
        unique_mom.push_back(q.mom);
      if(std::find(unique_disp.begin(), unique_disp.end(), q.displacement) == unique_disp.end())
        unique_disp.push_back(q.displacement);
      if(std::find(unique_gamma.begin(), unique_gamma.end(), q.gamma) == unique_gamma.end())
        unique_gamma.push_back(q.gamma);
    }

  file << "nx " << lat.nx << endl;
  file << "ny " << lat.ny << endl;
  file << "nz " << lat.nz << endl;
  file << "nt " << lat.nt << endl;
  file << "nvec 10" << endl;
  file << "cfg 100" << endl;
  file << "ndiags " << need_to_compute.size() << endl;
  file << "latname CFG/c44_b5.3_k0.158_100" << endl;
  file << "unique_mom:length " << unique_mom.size() << endl;
  for(size_t i=0; i<unique_mom.size(); ++i)
    file << "unique_mom:" << i << " " << unique_mom[i] << endl;

  file << "unique_gammas:length " << unique_gamma.size() << endl;
  for(size_t i=0; i<unique_gamma.size(); ++i)
    file << "unique_gammas:" << i << " " << unique_gamma[i] << endl;

  file << "unique_displacement:length " << unique_disp.size() << endl;
  for(size_t i=0; i<unique_disp.size(); ++i)
    file << "unique_displacement:" << i << " " << unique_disp[i] << endl;

	file << "dt_list ";
	for(size_t i=0; i<dts.size(); ++i)
	{
		file << dts[i];
		if(i != dts.size()-1 )
		 file << ",";
	}
	file << endl;
	file << "t_list ";
	for(size_t i=0; i<ts.size(); ++i)
	{
		file << ts[i];
		if(i != ts.size()-1 )
		 file << ",";
	}
	file << endl;
}


void Manager::compute_time_average_correlators()
{
	for(auto &c: corrs)
		c.compute_time_average_correlators();
}

void Manager::print_correlators()
{
	for(size_t i=0; i<ops.size(); ++i)
	for(size_t j=0; j<ops.size(); ++j)
	{
		ofstream corr_file;
		corr_file.open("corr_op."+to_string(i)+"_op."+to_string(j)+"_"+cfg_to_string(lat.cfg)+".dat");
		for(const auto &t : corrs[i*ops.size() + j].corr_t)
			corr_file << t.real() << " " << t.imag() << endl;
	}
}

void Manager::shutdown()
{
	auto main_logger = spdlog::get("main");
	main_logger->info("Program done");
	spdlog::shutdown();
}




void Manager::gpu_code_output(ofstream &cppfile, ofstream &gpufile, vector<Trace<QuarkLine>> need_to_compute)
{
	auto main_logger = spdlog::get("main");
	main_logger->info("Generating gpu code to compute traces");
	vector<string> unique_mom, unique_disp, unique_gamma;
	for(const auto& t : need_to_compute)
		for(const auto& q : t.qls)
		{
			if(find(unique_mom.begin(), unique_mom.end(), q.mom) == unique_mom.end())
				unique_mom.push_back(q.mom);
			if(find(unique_disp.begin(), unique_disp.end(), q.displacement) == unique_disp.end())
				unique_disp.push_back(q.displacement);
			if(find(unique_gamma.begin(), unique_gamma.end(), q.gamma) == unique_gamma.end())
				unique_gamma.push_back(q.gamma);
		}

	int max_size=0;
	for(auto t : need_to_compute)
		if(t.qls.size() > max_size)
			max_size=t.qls.size();
	///collect diagrams of certain lengths 1,2,3,4,5,6,etc
	vector<int> idx_track(max_size,0);///tracks idx for each length
	vector<int> res_idx;///for the result to pull the correct trace
	vector<vector<Trace_Code_Data>> traces_by_size(max_size);
	vector<vector<int>> orig_idx(max_size);
	//for(auto t : tr)
	for(size_t i=0; i<need_to_compute.size(); ++i)
	{
		auto t_size = need_to_compute[i].qls.size()-1;
		traces_by_size[t_size].push_back(
			Trace_Code_Data(need_to_compute[i].compute_name(unique_mom, unique_disp, unique_gamma), idx_track[t_size])
																		);

		orig_idx[t_size].push_back(i);
		res_idx.push_back(idx_track[t_size]);
		idx_track[t_size]++;
	}

	cppfile << gpu_code_cpp_prefix();
	gpufile << gpu_code_cuda_prefix();

	long int num_res_mats = need_to_compute.size();



	for(size_t l=0; l<traces_by_size.size(); ++l)
	{

		if(l==0)
			cppfile << "std::complex<double> *res = (std::complex<double> *)malloc(((long int)sizeof(std::complex<double>))*"
							<< "((long int)dim*dim)*((long int)" << traces_by_size[l].size() << "));\n";
		//else

		///Just take the trace of a quark line.
		if(l==0)
		{
			auto trs = traces_by_size[l];
			for(size_t d=0; d<trs.size(); ++d)
			{
				auto split_name = split(trs[d].compute_name[0], '[');
				cppfile << "memcpy(res + " << d << "*dim*dim, " << split_name[0]
				        << "s[" << split_name[1] << ", mat_size);";


				cppfile << endl;
			}
		}

		else
		{
			long int mat_size = 4*4*100*100*2*8; ///NS*NS*NDIM*NDIM*complex_double
			long int ql_mem = unique_mom.size()*unique_disp.size()*unique_gamma.size()*4*mat_size;
			long int num_sub_batches = traces_by_size[l].size()*3*mat_size/(gpu_memory*((long int)1000*1000) - ql_mem - ((long int)100*1000*1000) );
			cout << "num_sub_batches = " << num_sub_batches+1 << endl;
			vector<vector<Trace_Code_Data>> sub_batches(num_sub_batches+1);
			long int mem_used = ql_mem;
			int curr_list = 0;

			for(size_t d=0; d<traces_by_size[l].size(); ++d)
			{
				mem_used += ((long int)3)*mat_size;
				if( (gpu_memory*((long int)1000*1000)-mem_used) < ((long int)1000*1000*100) )
				{
			 		curr_list++;
					mem_used = ql_mem;
				}
				sub_batches[curr_list].push_back( traces_by_size[l][d] );
			}



			for(size_t s=0; s<sub_batches.size(); ++s)
			{
				if(l!=0)
					gpufile << gpu_code_function_prefix(to_string(l),to_string(s));
				auto trs = sub_batches[s];
				if(l!=0)
				{
			cppfile << "res = (std::complex<double> *)malloc(((long int)sizeof(std::complex<double>))*"
							<< "((long int)dim*dim)*((long int)" << trs.size() << "));\n";

				}
				//num_res_mats += trs.size();

				///just need to take the trace - do this on cpu.
				///First implementation assumes infinite memory in GPU.
				//cppfile << "batch_size[" << l << "] = " << trs.size() << ";" << endl;

				char arraylabel('A'), res_label('C');
				for(size_t i=0; i<3; ++i)
				{
					gpufile << "cudaMalloc((void **) &d_" << arraylabel << ", ((long int)" << sub_batches[s].size() << ")*mat_size);\n";
					arraylabel++;
				}

				vector<string> memcpy_qls(l+1, "");///hold the cudaMemcpy's so they can be placed appropriately

				for(size_t d=0; d<trs.size(); ++d)
				{
					auto s = trs[d].compute_name;
					vector<vector<string>> q;
					for(const auto e : s)
						q.push_back( split(e,'[') );

					for(auto &e : q)
						e[1].erase( remove(e[1].begin(), e[1].end(), ']'), e[1].end() );

					for(size_t i=0; i<=l; ++i)
					{
						arraylabel='A';
						if(i==1)
							arraylabel = 'B'; //only for the startup do I need to load into another array

							memcpy_qls[i] = memcpy_qls[i] + "cudaMemcpy(d_" + arraylabel + " + " + to_string(d)
								  	+ "*dim*dim, d_" + q[i][0] + " + (" + q[i][1]
										+ ")*dim*dim, mat_size, cudaMemcpyDeviceToDevice);\n";
					}
				}

				gpufile << memcpy_qls[0] << memcpy_qls[1];
				arraylabel='A';
				///First multiply uses the first two matrices in order.
				gpufile << "cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, "
								<< "dim, dim, dim, "
								<< "_alpha, d_" << arraylabel << ", dim, dim*dim, ";
				arraylabel++;
				gpufile << "d_" << arraylabel << ", dim, dim*dim, "
								<< "_beta, d_" << res_label << ", dim, dim*dim, "
								<< trs.size() << ");\n";
			  arraylabel='A';
				//The rest require the previous result.
				for(size_t i=1; i<l; ++i)
				{
					gpufile << memcpy_qls[i+1];
					gpufile << "cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, "
									<< "dim, dim, dim, "
									<< "_alpha, d_" << res_label << ", dim, dim*dim, ";

				  if(res_label == 'C')
						res_label = 'B';
				  else
					  res_label = 'C';

						gpufile << "d_" << arraylabel << ", dim, dim*dim, "
									<< "_beta, d_" << res_label << ", dim, dim*dim, "
									<< trs.size() << ");\n";
				}

				///offsets the memcpy from Device to Host
				long int res_offset = 0;
				for(size_t i=0; i<l; ++i)
					res_offset += traces_by_size[i].size();
				for(size_t i=0; i<s; ++i)
					res_offset += sub_batches[i].size();
				//if( (l!=1) && (s!=0))
				//	res_offset++;

				gpufile << "cudaMemcpy(res, d_" << res_label << ", " << trs.size() << "*mat_size, cudaMemcpyDeviceToHost);\n";

				arraylabel = 'A';
				for(size_t i=0; i<3; ++i)
				{
					gpufile << "cudaFree(d_" << arraylabel << ");\n";
					arraylabel++;
				}
			if(l!=0)
				cppfile << "cublas_batch_multiply_" << l << "_" << s << "(res, qfs, qbs, qtis, qtfs, dim);\n";

			///Now take the trace and free up memory for rest of calculation.
			cppfile << "for(int i=0; i<dim; ++i)\n"
							<< "{\n";
			for(size_t c=0; c<trs.size(); ++c)
			{
				int sub_offset = 0;
				for(int o=s-1; o>=0; o--)
					sub_offset += sub_batches[o].size();
				cppfile << "diag[" << orig_idx[l][c+sub_offset] << "][dt][t] += res[((long int)" << c << ")*dim*dim + i*dim + i];\n";
			}
			cppfile << "}\n";

			cppfile << "free(res);\n";
			if(l!=0)
				gpufile << gpu_code_function_postfix();

			}///end s loop


		}///end l>0 loop

	if(l==0)
	{
			///Now take the trace and free up memory for rest of calculation.
			cppfile << "for(int i=0; i<dim; ++i)\n"
							<< "{\n";
			for(size_t c=0; c<traces_by_size[l].size(); ++c)
			{
				cppfile << "diag[" << orig_idx[l][c] << "][dt][t] += res[((long int)" << c << ")*dim*dim + i*dim + i];\n";
			}
			cppfile << "}\n";

			cppfile << "free(res);\n";

	}

	}///end l loop





	cppfile << "for(auto e : qfs)\n"
					<< "	delete e;\n";
	cppfile << "for(auto e : qbs)\n"
					<< "	delete e;\n";
	cppfile << "for(auto e : qtis)\n"
					<< "	delete e;\n";
	cppfile << "for(auto e : qtfs)\n"
					<< "	delete e;\n";

	cppfile << "qfs.clear();\n";
	cppfile << "qbs.clear();\n";
	cppfile << "qtis.clear();\n";
	cppfile << "qtfs.clear();\n";





	cppfile << gpu_code_cpp_postfix();
	main_logger->info("GPU code output");
}









/// The above CPU code generation produces computations like d6[0] = res3*res3
/// AND 																										 d6[1] = ql*res4*ql
/// Which would make for some complicated GPU code generation logic with the
/// batched matrix multiply routine.

/// For a first effort I will just replace ql*ql with res2.  Never subbing in a
/// res 3 or 4.

///


/*
void Manager::gpu_code_output(ofstream &cppfile, ofstream &gpufile, vector<Trace> need_to_compute)
{
	gpufile << gpu_code_cuda_prefix();

	vector<string> unique_mom, unique_disp, unique_gamma;
	for(const auto& t : need_to_compute)
		for(const auto& q : t.qls)
		{
			if(find(unique_mom.begin(), unique_mom.end(), q.mom) == unique_mom.end())
				unique_mom.push_back(q.mom);
			if(find(unique_disp.begin(), unique_disp.end(), q.displacement) == unique_disp.end())
				unique_disp.push_back(q.displacement);
			if(find(unique_gamma.begin(), unique_gamma.end(), q.gamma) == unique_gamma.end())
				unique_gamma.push_back(q.gamma);
		}

	int max_size=0;
	for(auto t : need_to_compute)
		if(t.qls.size() > max_size)
			max_size=t.qls.size();
	///collect diagrams of certain lengths 1,2,3,4,5,6,etc
	vector<int> idx_track(max_size,0);///tracks idx for each length
	vector<int> res_idx;///for the result to pull the correct trace
	vector<vector<Trace_Code_Data>> traces_by_size(max_size);
	//for(auto t : tr)
	for(size_t i=0; i<need_to_compute.size(); ++i)
	{
		auto t_size = need_to_compute[i].qls.size()-1;
		traces_by_size[t_size].push_back(
			Trace_Code_Data(need_to_compute[i].compute_name(unique_mom, unique_disp, unique_gamma), idx_track[t_size])
																		);
		res_idx.push_back(idx_track[t_size]);
		idx_track[t_size]++;
	}

	cppfile << gpu_code_cpp_prefix();

	long int num_res_mats = need_to_compute.size();
	cppfile << "int dim = qf[0].rows();\n";

	cppfile << "std::complex<double> *res = (std::complex<double> *)malloc(((long int)sizeof(std::complex<double>))*"
	        << "((long int)dim*dim)*((long int)" << to_string(num_res_mats) << "));\n";
	cppfile << "vector<std::complex<double>*> qfs, qbs, qtis, qtfs;\n";
	cppfile << "for(size_t i=0; i<qf.size(); ++i)\n"
					<< "{\n"
					<< "qfs.push_back(new std::complex<double>[dim*dim]);\n"
					<< "qbs.push_back(new std::complex<double>[dim*dim]);\n"
					<< "qtis.push_back(new std::complex<double>[dim*dim]);\n"
					<< "qtfs.push_back(new std::complex<double>[dim*dim]);\n"
					<< "}\n";

	cppfile << "for(size_t i=0; i<qf.size(); ++i)\n"
					<< "{\n"
					<< "Eigen::Map<mat>(qfs[i],dim,dim) = qf[i];\n"
					<< "Eigen::Map<mat>(qbs[i],dim,dim) = qb[i];\n"
					<< "Eigen::Map<mat>(qtis[i],dim,dim) = qti[i];\n"
					<< "Eigen::Map<mat>(qtfs[i],dim,dim) = qtf[i];\n"
					<< "}\n";


	gpufile << "cuDoubleComplex *d_A, *d_B, *d_C;\n";

	for(size_t l=0; l<traces_by_size.size(); ++l)
	{
		///Just take the trace of a quark line.
		if(l==0)
		{
			auto trs = traces_by_size[l];
			cppfile << "long int mat_size = dim*dim*sizeof(std::complex<double>);\n";
			for(size_t d=0; d<trs.size(); ++d)
			{
				auto split_name = split(trs[d].compute_name[0], '[');
				cppfile << "memcpy(res + " << d << "*dim*dim, " << split_name[0]
				        << "s[" << split_name[1] << ", mat_size);";


				cppfile << endl;
			}
		}
		else
		{
			long int mat_size = 4*4*100*100*2*8; ///NS*NS*NDIM*NDIM*complex_double
			long int ql_mem = unique_mom.size()*unique_disp.size()*unique_gamma.size()*4*mat_size;
			long int num_sub_batches = traces_by_size[l].size()*3*mat_size/(gpu_memory*((long int)1000*1000) - ql_mem - ((long int)100*1000*1000) );
			cout << "num_sub_batches = " << num_sub_batches+1 << endl;
			vector<vector<Trace_Code_Data>> sub_batches(num_sub_batches+1);
			long int mem_used = ql_mem;
			int curr_list = 0;

			for(size_t d=0; d<traces_by_size[l].size(); ++d)
			{
				mem_used += ((long int)3)*mat_size;
				if( (gpu_memory*((long int)1000*1000)-mem_used) < ((long int)1000*1000*100) )
				{
			 		curr_list++;
					mem_used = ql_mem;
				}
				sub_batches[curr_list].push_back( traces_by_size[l][d] );
			}



			for(size_t s=0; s<sub_batches.size(); ++s)
			{
				auto trs = sub_batches[s];
				//num_res_mats += trs.size();

				///just need to take the trace - do this on cpu.
				///First implementation assumes infinite memory in GPU.
				//cppfile << "batch_size[" << l << "] = " << trs.size() << ";" << endl;

				char arraylabel('A'), res_label('C');
				for(size_t i=0; i<3; ++i)
				{
					gpufile << "cudaMalloc((void **) &d_" << arraylabel << ", ((long int)" << sub_batches[s].size() << ")*mat_size);\n";
					arraylabel++;
				}

				vector<string> memcpy_qls(l+1, "");///hold the cudaMemcpy's so they can be placed appropriately

				for(size_t d=0; d<trs.size(); ++d)
				{
					auto s = trs[d].compute_name;
					vector<vector<string>> q;
					for(const auto e : s)
						q.push_back( split(e,'[') );

					for(auto &e : q)
						e[1].erase( remove(e[1].begin(), e[1].end(), ']'), e[1].end() );

					for(size_t i=0; i<=l; ++i)
					{
						arraylabel='A';
						if(i==1)
							arraylabel = 'B'; //only for the startup do I need to load into another array

							memcpy_qls[i] = memcpy_qls[i] + "cudaMemcpy(d_" + arraylabel + " + " + to_string(d)
								  	+ "*dim*dim, d_" + q[i][0] + " + (" + q[i][1]
										+ ")*dim*dim, mat_size, cudaMemcpyDeviceToDevice);\n";
					}
				}

				gpufile << memcpy_qls[0] << memcpy_qls[1];
				arraylabel='A';
				///First multiply uses the first two matrices in order.
				gpufile << "cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, "
								<< "dim, dim, dim, "
								<< "_alpha, d_" << arraylabel << ", dim, dim*dim, ";
				arraylabel++;
				gpufile << "d_" << arraylabel << ", dim, dim*dim, "
								<< "_beta, d_" << res_label << ", dim, dim*dim, "
								<< trs.size() << ");\n";
			  arraylabel='A';
				//The rest require the previous result.
				for(size_t i=1; i<l; ++i)
				{
					gpufile << memcpy_qls[i+1];
					gpufile << "cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, "
									<< "dim, dim, dim, "
									<< "_alpha, d_" << res_label << ", dim, dim*dim, ";

				  if(res_label == 'C')
						res_label = 'B';
				  else
					  res_label = 'C';

						gpufile << "d_" << arraylabel << ", dim, dim*dim, "
									<< "_beta, d_" << res_label << ", dim, dim*dim, "
									<< trs.size() << ");\n";
				}

				///offsets the memcpy from Device to Host
				long int res_offset = 0;
				for(size_t i=0; i<l; ++i)
					res_offset += traces_by_size[i].size();
				for(size_t i=0; i<s; ++i)
					res_offset += sub_batches[i].size();
				//if( (l!=1) && (s!=0))
				//	res_offset++;

				gpufile << "cudaMemcpy(res + ((long int)" << res_offset << ")*dim*dim, d_" << res_label << ", " << trs.size() << "*mat_size, cudaMemcpyDeviceToHost);\n";

				arraylabel = 'A';
				for(size_t i=0; i<3; ++i)
				{
					gpufile << "cudaFree(d_" << arraylabel << ");\n";
					arraylabel++;
				}

			}///end if (l>0)

		}///end s loop

	}///end l loop



	cppfile << "cublas_batch_multiply_all(res, qfs, qbs, qtis, qtfs, dim);\n";

	cppfile << "for(auto e : qfs)\n"
					<< "	delete e;\n";
	cppfile << "for(auto e : qbs)\n"
					<< "	delete e;\n";
	cppfile << "for(auto e : qtis)\n"
					<< "	delete e;\n";
	cppfile << "for(auto e : qtfs)\n"
					<< "	delete e;\n";

	cppfile << "qfs.clear();\n";
	cppfile << "qbs.clear();\n";
	cppfile << "qtis.clear();\n";
	cppfile << "qtfs.clear();\n";


	gpufile << gpu_code_cuda_postfix();

	///Now finish off the cpu code
	cppfile << "for(int i=0; i<dim; ++i)\n"
					<< "{\n";
	for(size_t d=0; d<need_to_compute.size(); ++d)
	{
		int d_idx = res_idx[d];
		int l_idx = 0;
		///Also need to adjust it to the right batch location.
		for(size_t l=0; l<traces_by_size.size(); ++l)
      if(need_to_compute[d].qls.size()==(l+1))
				l_idx = l;

		for(int l=0; l<l_idx; ++l)
			d_idx += traces_by_size[l].size();

		cppfile << "diag[" << d << "][dt][t] += res[((long int)" << d_idx << ")*dim*dim + i*dim + i];\n";
	}
	cppfile << "}\n";

	cppfile << gpu_code_cpp_postfix();

}
*/
