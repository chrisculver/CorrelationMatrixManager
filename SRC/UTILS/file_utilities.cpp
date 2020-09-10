#include "UTILS/file_utilities.h"
#include "UTILS/string_utilities.h"
#include "manager.h"

#include <fstream>

using namespace std;
using Saved_Diagrams = map<string, map<string,complex<double>>>;

vector<string> load_diagram_filenames(string file, string cfg)
{
  ifstream input(file);
  string line;
  vector<string> res;
  while(getline(input,line))
    res.push_back(line+cfg+".dat");

  input.close();
  return res;
}

bool file_exists(string filename)
{
  ifstream file(filename.c_str());
  return file.good();
}

Saved_Diagrams parse_diagram_file(std::string filename)
{
	ifstream input(filename);
  string line;
  string current_diagram;
  Saved_Diagrams res;

  while(getline(input, line))
  {
    if(line[0]=='[')
      current_diagram=line;
    else
    {
      auto columns = split(line, ' ');
      res[current_diagram][columns[0]+" "+columns[1]]=std::complex<double>{stod(columns[2]), stod(columns[3])};
    }
  }
  input.close();

//  Saved_Traces computed_traces;
//  for(size_t i=0; i<all_names.size(); ++i)
//    computed_traces[all_names[i]] = all_values[i];

  return res;

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
