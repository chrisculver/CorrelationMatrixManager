// control_clover-prop_2.cu
// Dehua Guo
// Dec. 2013

#include <stdlib.h>
#include <time.h> // for timing
#include "terminate.h" // Error messages and exit functions 
#include "layout.h"
#include "layout_minsurface.h"
#include "inverters_device.h"
#include "site.h"
#include "options.h"
#include "generic_qcd_fields.h"
#include "timer.h"
#include "boundary_flip.h"
#include "io_kentucky.h"
#include "field_rotate.h"
#include "clover_multi_cpu.h"
#include "clover_multi_gpu.h"
#include "clover_prec_state_device.h"
#include "vector_util.h"
#include "smearing.h"
#include <vector>
#include "comm/comm_low.h"

using namespace qcd;
#define print0 if( get_node_rank() == 0) printf

//void read_prop(std::string &name, vector** prop);

int main(int argc, char* argv[])
{
  //start mpi
  init_machine(argc, argv);

  //input parameters 
  int max_cg_iterations;
  double error_cg;
  int nx, ny, nz, nt;
  int point_source_position_x;
  int point_source_position_y;
  int point_source_position_z;
  int point_source_position_t;
  int Nmasses;
  std::vector<double> kappa;
  textline bcline;
  std::string flinks, fprop;

  options opts;
  opts.add ("nx",nx);
  opts.add ("ny",ny);
  opts.add ("nz",nz);
  opts.add ("nt",nt);
  opts.add ("boundary_conditions", bcline); // 1 1 1 -1
  opts.add ("kappa",kappa);
  opts.add ("num_kappas", Nmasses);
  opts.add ("config",flinks);
  opts.add ("point_source_position_x", point_source_position_x);
  opts.add ("point_source_position_y", point_source_position_y);
  opts.add ("point_source_position_z", point_source_position_z);
  opts.add ("point_source_position_t", point_source_position_t);
  opts.add ("max_cg_iterations", max_cg_iterations);
  opts.add ("error_cg", error_cg);
  opts.read_options(argc,argv);

  //setup the lattice 
  layout_minsurface_eo desc(nx,ny,nz,nt);
  su3_field origlinks(&desc);
  su3_field links(&desc);
  vector vecTemp(&desc);

  //read in links and fields etc..
  if(get_node_rank()==0) printf("HEREREERERER\n");
  FILE *f;
  if(get_node_rank() == 0)
  {
    f = fopen(flinks.c_str(), "rb");
    if(f==NULL)
    {
      printf("Error: opening lattice file %s failed. Exiting ...\n", flinks.c_str());
      terminate_error(ERROR_IO_ACCESS);
    }
  }
  //parallel_io io(1, 200);
  read_kentucky_lattice(f,origlinks);
  if(get_node_rank()==0) printf("smearing links\n");
  nhyp_smear_simple(origlinks, links);

  //set up boundary condtions
  int bc[4];
  {
    std::stringstream ss(bcline);
    ss >> bc[0] >> bc[1] >> bc[2] >> bc[3];
    for(int i=0; i<4; ++i) if(bc[i]>1 || bc[i]<-1)
    {
      print0("Incorrect boundary conditions ... exiting\n");
      return terminate_error(ERROR_READING_EXTENDED_SET);
    }
    print0("Boundary conditions: xyzt -> %+d %+d %+d %+d\n", bc[0], bc[1], bc[2], bc[3]);
  } 

  apply_boundary(links, bc);

  // create instance of dslash  
  double *kappa_darray = &kappa[0];
  qcd::prec_clover_device pcd(links, kappa_darray, Nmasses, 1);
  qcd::clover_state hclover(links);

  //create vectors and solutions for inversions
  //qcd::device_wilson_field srcDev(&desc);
  qcd::device_wilson_field solDev(&desc);
  vector* solutions[Nmasses];
  for(int i=0; i< Nmasses; i++) solutions[i] = new vector(&desc);
  vector* qprop[12*Nmasses];
  for(int i=0; i<12*Nmasses; i++) qprop[i] = new vector(&desc);

  for(int s=0; s<4; ++s)
    for(int c=0; c<3; ++c)
    {
      // setup the delta source
      position p(point_source_position_x, point_source_position_y,
	  point_source_position_z, point_source_position_t);
      int max_iter = max_cg_iterations;
      double error = error_cg;

      clear_vector(vecTemp);
      int idx = (&desc)->get_site_index(p);

      if(get_node_rank() == (&desc)->get_site_rank(p))
	vecTemp.data[idx].d[s].c[c] = 1.0;
      kentucky2milc_wilsonfield(vecTemp);

      //perform inversion
      qcd::clover_multimass_device_eo(pcd, vecTemp, solutions,error, max_iter);

      // Compute residues
      vector vecTemp2(&desc);
      for(int i=0; i<Nmasses; ++i)
      {
	// boost the solutions to get results similar to the cpu code
        *solutions[i] = 2*kappa[i]*(*solutions[i]);
	hclover.clover(*solutions[i], vecTemp2, kappa[i], 1.0);
        vecTemp2 = vecTemp + (-1.0/(2*kappa[i]))*vecTemp2;

	//double nrm = norm(vecTemp2)/norm(vecTemp);
	double nrm = norm(vecTemp2);
	if(get_node_rank() == 0) printf("residue for kappa=%0.5f is %15.10le\n", kappa[i], nrm);
	if ((nrm > 1.5*error_cg) || isnan(nrm))
	{
	  if(get_node_rank() == 0) printf("Inversion error encountered ... exiting.\n");
	  exit(1);
	}

	milc2kentucky_wilsonfield(*solutions[i]);
	copy_vec1_to_vec2(*solutions[i], *qprop[12*i + s*3+ c]);
      }
    }
 double_complex Gt[nt];
  for(int i=0; i<nt; i++) Gt[i] = 0.0;
  position p;
  for(int src = 0; src<12; src++)
    for(int n = 0; n < desc.sites_on_node; n++)
      for(int d = 0; d<4; d++)
	for(int c = 0; c<3; c++)
	{
	  p = desc.get_position(n, get_node_rank());
	  Gt[p[3]] += conj(qprop[src]->data[n].d[d].c[c])*(qprop[src]->data[n].d[d].c[c]);
	}

  for(int i=0; i < nt; i++) global_sum(Gt[i]);

  for(int i=0; i < nt; i++)
    if(get_node_rank() == 0)
    {
      printf("Gt %02d %e\n",i,Gt[i].real);
    }



  delete solutions[0];
  for(int i=0; i<12; ++i) delete qprop[i];

  shutdown_machine();
  return 0;
}
