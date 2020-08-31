#include "UTILS/file_utilities.h"
#include "UTILS/string_utilities.h"
#include "manager.h"

#include <fstream>

using namespace std;
using Saved_Traces = map<string, vector<vector<complex<double>>> >;

bool file_exists(string filename)
{
  ifstream file(filename.c_str());
  return file.good();
}

Saved_Traces parse_diagram_file(std::string filename, int NT)
{
	ifstream input(filename);
  string line;
  vector<string> all_names;
  vector<vector<vector<complex<double>>>> all_values;
  while(getline(input, line))
  {
		vector<vector<complex<double>>> tmp_val(NT);
    /// for dt in dts
    /// resize( length(ts) )
    for(int t=0; t<NT; ++t)
      tmp_val[t].resize(NT);

    if(line[0]=='[')
      all_names.push_back(line);
    else
    {


      auto idx=0;
      auto vals=split(line, ' ');
      for(int i=0; i<NT; ++i)
        tmp_val[idx][i] = std::complex<double>{stod(vals[2*i]), stod(vals[2*i+1])};
      idx++;
      for(int j=1; j<NT; ++j)
      {
        getline(input, line);
        auto vals = split(line, ' ');
        for(int i=0; i<NT; ++i)
          tmp_val[idx][i] = std::complex<double>{stod(vals[2*i]), stod(vals[2*i+1])};
        idx++;
      }
      all_values.push_back(tmp_val);
    }
  }
  input.close();

  Saved_Traces computed_traces;
  for(size_t i=0; i<all_names.size(); ++i)
    computed_traces[all_names[i]] = all_values[i];

  return computed_traces;

}

string cpp_prefix()
{
  string text;

  text += "#include \"define_diagrams.h\"\n";
	text += "typedef Eigen::MatrixXcd mat;\n";
	text += "using namespace std;\n";
	text += "void define_diagrams(vector< vector< vector < complex<double> > > > &diag, const vector<mat> qf, const vector<mat> qb,";
  text += "const vector<mat> qti, const vector<mat> qtf, int t, int dt)\n";
	text += "{\n";
  return text;
}

string cpp_postfix()
{
  return "\n}";
}

string gpu_code_cpp_prefix()
{
  string text;
  text += "#include \"define_diagrams.h\"\n";
  text += "#include \"SRC/gpu_kernel.h\"\n";
  text += "#include <complex>\n";
  text += "typedef Eigen::MatrixXcd mat;\n";
  text += "using namespace std;\n";
  text += "void define_diagrams(vector< vector< vector < complex<double> > > > &diag, const vector<mat> qf, const vector<mat> qb,";
  text += "const vector<mat> qti, const vector<mat> qtf, int t, int dt)\n";
  text += "{\n";
  text += "int dim = qf[0].rows();\n";
  text += "vector<std::complex<double>*> qfs, qbs, qtis, qtfs;\n";
	text += "for(size_t i=0; i<qf.size(); ++i)\n";
	text += "{\n";
	text += "qfs.push_back(new std::complex<double>[dim*dim]);\n";
	text += "qbs.push_back(new std::complex<double>[dim*dim]);\n";
	text += "qtis.push_back(new std::complex<double>[dim*dim]);\n";
	text += "qtfs.push_back(new std::complex<double>[dim*dim]);\n";
	text += "}\n";

	text += "for(size_t i=0; i<qf.size(); ++i)\n";
	text += "{\n";
	text += "Eigen::Map<mat>(qfs[i],dim,dim) = qf[i];\n";
	text += "Eigen::Map<mat>(qbs[i],dim,dim) = qb[i];\n";
	text += "Eigen::Map<mat>(qtis[i],dim,dim) = qti[i];\n";
	text += "Eigen::Map<mat>(qtfs[i],dim,dim) = qtf[i];\n";
	text += "}\n";
  text += "long int mat_size = dim*dim*sizeof(std::complex<double>);\n";
  return text;
}

string gpu_code_cuda_prefix()
{
  string text;
  text += "#include \"gpu_kernel.h\"\n";
  text += "#include <cuda_runtime.h>\n";
  text += "#include <cublas_v2.h>\n\n";
  return text;
}

string gpu_code_function_prefix(std::string l, std::string s)
{
  string text;

  text = text + "void cublas_batch_multiply_" + l + "_" + s + "(std::complex<double> *res,"
                                          +"std::vector<std::complex<double>*> qf,"
                                          +"std::vector<std::complex<double>*> qb,"
                                          +"std::vector<std::complex<double>*> qti,"
                                          +"std::vector<std::complex<double>*> qtf,"
                                          +"long int dim"
                                          +")\n {\n";
  text = text +      "cudaError_t cudaStat;\n"
	      +     "cublasStatus_t stat;\n"
	      +     "cublasHandle_t handle;\n"
	      +     "cudaStream_t stream;\n"
	      +     "cudaStreamCreate(&stream);\n"
	      +     "stat = cublasCreate(&handle);\n"
	      +     "cublasSetStream(handle, stream);\n"
	      +     "std::complex<double> alpha(1.,0.);\n"
	      +     "std::complex<double> beta(0.,0.);\n"
	      +     "cuDoubleComplex *_alpha = reinterpret_cast<cuDoubleComplex*>(&alpha);\n"
	      +     "cuDoubleComplex *_beta = reinterpret_cast<cuDoubleComplex*>(&beta);\n"
	      +     "int block_size = 32;\n"
	      +     "dim3 threads(block_size, block_size);\n"
	      +     "dim3 grid(dim/threads.x, dim/threads.y);\n"
        +     "long int mat_size = dim*dim*sizeof(std::complex<double>);\n";

  text = text + "cuDoubleComplex *d_qf, *d_qb, *d_qti, *d_qtf;\n"
              + "cudaMalloc((void **) &d_qf, qf.size()*mat_size);\n"
              + "cudaMalloc((void **) &d_qb, qb.size()*mat_size);\n"
              + "cudaMalloc((void **) &d_qti, qti.size()*mat_size);\n"
              + "cudaMalloc((void **) &d_qtf, qtf.size()*mat_size);\n";

  text = text + "for(int i=0; i<qf.size(); ++i)\n"
              + "{\n"
              + "cudaMemcpy(d_qf + i*dim*dim, qf[i], mat_size, cudaMemcpyHostToDevice);\n"
              + "cudaMemcpy(d_qb + i*dim*dim, qb[i], mat_size, cudaMemcpyHostToDevice);\n"
              + "cudaMemcpy(d_qti + i*dim*dim, qti[i], mat_size, cudaMemcpyHostToDevice);\n"
              + "cudaMemcpy(d_qtf + i*dim*dim, qtf[i], mat_size, cudaMemcpyHostToDevice);\n"
              + "}\n";
  text += "cuDoubleComplex *d_A, *d_B, *d_C;\n";



  return text;
}

string gpu_code_function_postfix()
{
  string text;

  text += "cudaFree(d_qf);\n";
  text += "cudaFree(d_qb);\n";
  text += "cudaFree(d_qtf);\n";
  text += "cudaFree(d_qti);\n";

  text += "cublasDestroy(handle);\n";
  text += "cudaStreamSynchronize(0);\n";
  text += "cudaStreamDestroy(stream);\n";
  text += "}";

  return text;
}

string gpu_code_cpp_postfix()
{
  string text;
  text += "}";
  return text;
}
