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

	files = FileNames(name_value["operator_filename"], "diags_"+to_string(lat.nx)+to_string(lat.ny)
				+to_string(lat.nz)+to_string(lat.nt)+"_"+cfg_to_string(lat.cfg)+".dat");
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

	for(const auto &c : ops)
		for(const auto &a : ops)
			corrs.push_back( Correlator(adjoint(a), c) );


	op_logger->flush();
	main_logger->flush();
}


void Manager::wick_contractions()
{
//	for(auto &c: corrs)
//		c.wick_contract();
	for(size_t i=0; i<ops.size(); ++i)
	for(size_t j=0; j<ops.size(); ++j)
	{
	//	cout << "wick contraction for c_ij = " << i << " " << j << endl;
		corrs[i*ops.size() +j].wick_contract();
		auto wick_logger = spdlog::get("wick");
		std::string diag_names;
		for(const auto &d: corrs[i*ops.size() + j].diags)
			diag_names+=d.name()+"\n";
		wick_logger->info("Resulting diags for c_{}{} = \n{}", i, j, diag_names);
	//	cout << endl;
	}


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
	if( !file_exists(files.diagram_filename) )
	{
		main_logger->info("No diagram file found");
		throw traces_to_compute();
	}

	auto computed = parse_diagram_file(files.diagram_filename, lat.nt);

	for(auto &c: corrs)
	{
		try
		{
			c.load_numerical_results(computed);
		}
		catch(char missing)
		{
			vector<string> computed_names;
			for(const auto &c : computed)
				computed_names.push_back(c.first);
			throw traces_to_compute(computed_names);
		}
	}
}


vector<Trace> Manager::traces_to_compute()
{
	vector<Trace> res;
	for(auto &c: corrs)
	for(auto &d: c.diags)
	for(auto &t: d.traces)
	{
		Trace r = t;
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

	return res;
}

vector<Trace> Manager::traces_to_compute(const vector<string> computed_names)
{
	///TODO : make this use computed_names, I don't want to do this yet until
	/// I have some safe way to merge/combine/handle diagram files better.
	vector<Trace> res;
	for(auto &c: corrs)
	for(auto &d: c.diags)
	for(auto &t: d.traces)
	{
		Trace r = t;
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

	return res;
}



void Manager::cpu_code_output(ofstream &file, vector<Trace> need_to_compute)
{
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
  ///traces_by_size


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
    }
    else
    {
      for(size_t d=0; d<trs.size(); ++d)
      {
        file << "res" << l+1 << "[" << d << "]=";

        vector<string> computation = trs[d].compute_name;
        bool max_cse=false;
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
            }
            if(max_cse)
              break;
          }
          if(max_cse)
            break;
        }
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

  for(size_t d=0; d<need_to_compute.size(); ++d)
    for(size_t l=0; l<traces_by_size.size(); ++l)
      if(need_to_compute[d].qls.size()==(l+1))
        file << "diag[" << d << "][dt][t] = res" << l+1 << "[" << res_idx[d] << "].trace();" << endl;

  file << cpp_postfix();

}

void Manager::diagram_names_output(ofstream &file, vector<Trace> need_to_compute)
{
	for(const auto &t: need_to_compute)
		file << t.name() << endl;
}


void Manager::runtime_input_for_cpu(ofstream &file, vector<Trace> need_to_compute)
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
}


void Manager::compute_time_average_correlators()
{
	for(auto &c: corrs)
		c.compute_time_average_correlators(lat.nt);
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
	spdlog::shutdown();
}



/// The above CPU code generation produces computations like d6[0] = res3*res3
/// AND 																										 d6[1] = ql*res4*ql
/// Which would make for some complicated GPU code generation logic with the
/// batched matrix multiply routine.

/// For a first effort I will just replace ql*ql with res2.  Never subbing in a
/// res 3 or 4.

///

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

	cppfile << gpu_code_cpp_prefix(traces_by_size.size());

	long int num_res_mats = 0;

	for(size_t l=0; l<traces_by_size.size(); ++l)
	{
		///just need to take the trace - do this on cpu.
		auto trs = traces_by_size[l];
		num_res_mats += trs.size();

		if(l==0)
		{
			for(size_t d=0; d<trs.size(); ++d)
			{
				cppfile << "res" << l+1 << "[" << d << "]=";
				//for(auto q : trs[d].compute_name)
				for(size_t i=0; i<trs[d].compute_name.size(); ++i)
				{
					cppfile << trs[d].compute_name[i];
					if(i!=trs[d].compute_name.size()-1)
						cppfile << "*";
					else
						cppfile << ";";
				}
				cppfile << endl;
			}
		}

		///do all the multiplications on gpu and save.
		if(l==1)
		{
			cppfile << "batch_size[" << l << "] = " << trs.size() << ";" << endl;

			gpufile << "cuDoubleComplex *d_twoA, *dtwoB, *d_twoC;\n";
			gpufile << "cudaMalloc((void **) &d_twoA, batch[" << l << "]*mat_size);\n";
			gpufile << "cudaMalloc((void **) &d_twoB, batch[" << l << "]*mat_size);\n";
			gpufile << "cudaMalloc((void **) &d_twoC, batch[" << l << "]*mat_size);\n";

			for(size_t d=0; d<trs.size(); ++d)
			{

				auto s = trs[d].compute_name;
				auto q0 = split(s[0],'[');
				auto q1 = split(s[1],'[');

				q0[1].erase ( remove(q0[1].begin(), q0[1].end(), ']'), q0[1].end() );
				q1[1].erase ( remove(q1[1].begin(), q1[1].end(), ']'), q1[1].end() );
				gpufile << "cudaMemcpy(d_twoA + " << to_string(d) << "*dim*dim, d_"
				         << q0[0] << " + " << q0[1]
								 << "*dim*dim, mat_size, cudaMemcpyDeviceToDevice);" << endl;
				gpufile << "cudaMemcpy(d_twoB + " << to_string(d) << "*dim*dim, d_"
 				         << q1[0] << " + " << q1[1]
 								 << "*dim*dim, mat_size, cudaMemcpyDeviceToDevice);" << endl;
			}

			gpufile << "cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, "
			        << "dim, dim, dim, "
							<< "_alpha, d_twoA, dim, dim*dim, d_twoB, dim, dim*dim, "
							<< "_beta, d_twoC, dim, dim*dim, batch[" << l << "]);\n";

			gpufile << "cudaMemcpy(res, d_twoC, batch[" << l << "]*mat_size, cudaMemcpyDeviceToHost);\n";
			gpufile << "cudaFree(d_twoA);\n";
			gpufile << "cudaFree(d_twoB);\n";
			gpufile << "cudaFree(d_twoC);\n";

		}

		///We now need to look for them in the above computation.
		if(l>1)
		{

		}

	}




	cppfile << "int dim = qf[0].rows();\n";

	cppfile << "std::complex<double> *res = (std::complex<double> *)malloc(sizeof(std::complex<double>)*"
	        << ")*dim*dim*" << to_string(num_res_mats) << ");\n";

	cppfile << "vector<std::complex<double>*> qfs, qbs, qtis, qtfs;\n";
	cppfile << "for(size_t i=0; i<qf.size(); ++i)\n"
					<< "{\n"
					<< "qfs.push_back(new std::cmplex<double>[dim*dim]);\n"
					<< "qbs.push_back(new std::cmplex<double>[dim*dim]);\n"
					<< "qtis.push_back(new std::cmplex<double>[dim*dim]);\n"
					<< "qtfs.push_back(new std::cmplex<double>[dim*dim]);\n"
					<< "}\n";

	cppfile << "for(size_t i=0; i<qf.size(); ++i)\n"
					<< "{\n"
					<< "Eigen::Map<mat>(qfs[i],dim,dim) = qf[i];\n"
					<< "Eigen::Map<mat>(qbs[i],dim,dim) = qb[i];\n"
					<< "Eigen::Map<mat>(qtis[i],dim,dim) = qti[i];\n"
					<< "Eigen::Map<mat>(qtfs[i],dim,dim) = qtf[i];\n"
					<< "}\n";

	cppfile << "cublas_batch_multiply_all(res, qfs, qbs, qtis, qtfs, dim, batch_size);\n";

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
		int d_idx = 0;
		cppfile << "diag[" << d << "][dt][t] = res[" << d_idx << "*dim*dim + i*dim + i];\n";
	}
	cppfile << "}\n";

	cppfile << gpu_code_cpp_postfix();

}
