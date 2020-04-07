// laplacean_eigensolver.cpp
// Dehua Guo
// Aug 2014

#include "laplacean_eigensolver.h"
#include "random.h"
#include "matmult.h"
#include "comm/comm_shift.h"
#include "layout_minsurface.h"
#include "inject.h"
#include "../eigensystem/arnoldi.h"
#include "eigensystem.h"
//#include "../eigensystem/arnoldi_template.h"
#include "../eigensystem/vec_eigen_pair_template.h"
#include "../eigensystem/eigensystem_util_template.h"
#include "../eigensystem/chebyshev_eigensystem_template.h"
#include "../eigensystem/chebyshev_generic_template.h"
#include "lapmult.h"
#include "../io/communicator.h"

namespace qcd{

template struct chebyshev_generic_matmult<colorvector>;

static int abssmaller(const double_complex& a, const double_complex& b)
{ 
    if (norm(a) >= norm(b)) return 1;
      else return -1;
} 


void check_vector_allocation(colorvector** p) {};

void laplacean_eigensolver::free_eigensystem()
{
//This is causing seg faults when mom matrices are 
//placed into functions.

/*  if(eig == NULL) return;
  for(int i=0; i<nt; ++i) 
			delete eig[i];
	delete [] eig;
*/
}

void laplacean_eigensolver::solve(int nvec, int order, double sigma)
{
  allocate_eigensystem(nvec);

  su3_field linksproj(&desc);

  lapmult lm(linksproj);
  colorvector start(&desc);
  random_field rnd(&desc);

  for(int t=0; t<nt; ++t)
  {
    extract(links, linksproj, position(0, 0, 0, t));
    random_vector_uniform(start, rnd);
    if(order == 0)
      arnoldi_eigensystem(lm, start, eig[t]->eval, eig[t]->evec, nvec, 2*nvec, 1e-10, 100000, abssmaller, double_complex(-10,0), SYM);
    else
    {
      eig[t]->resize(2*nvec);
      chebyshev_eigensystem(1e-10, 100000, order, -sigma, nvec, *eig[t], start, lm, -12); 
    eig[t]->resize(nvec);
    }
  }

}

void laplacean_eigensolver::allocate_eigensystem(int nvec)
{
  free_eigensystem();
  eig = new vec_eigen_pair<colorvector>*[nt];
  for(int t=0; t<nt; ++t) eig[t] = new vec_eigen_pair<colorvector>(nvec, desc);
}

void laplacean_eigensolver::check_eigensystem()
{
  su3_field linksproj(&desc);
  lapmult lm(linksproj);
  std::vector<double> resid(this->eigsys(0).size);
  for(int t=0; t<nt; ++t)
  {
    extract(links, linksproj, position(0, 0, 0, t));
    vec_eigen_pair<colorvector>& eig = this->eigsys(t);
    compute_residues(eig.eval, eig.evec, eig.size, lm, &resid[0], false);
  }
}

laplacean_eigensolver::~laplacean_eigensolver()
{
  free_eigensystem();
}

void save_laplace_eigensystem(const std::string& name, laplacean_eigensolver& l) 
{
  lattice_desc &desc = l.desc;
  int nvec = l.eigsys(0).size;
  parallel_io io(2);
  generic_communicator<> com(io, desc, sizeof(su3_vector)*l.nt, generic_communicator<>::write);
  
  FILE* f;
  f = io.fopen(name.c_str(), "wb");

  su3_vector* buf = (su3_vector*) com.comm_buffer;
  for(int v=0; v<nvec; ++v)
  {
    for(int i=0; i<desc.sites_on_node; ++i)
    for(int t=0; t<l.nt; ++t) buf[t+l.nt*i] = l.eigsys(t).evec[v]->data[i];
    com.execute(f);
  }

  if(get_node_rank() == io.io_nodes[0])
  {
    for(int t=0; t<l.nt; ++t) fwrite(l.eigsys(t).eval, sizeof(double_complex), nvec, f);
  }

  io.fclose(f);

}

void read_laplace_eigensystem(const std::string& name, laplacean_eigensolver& l) 
{
  lattice_desc &desc = l.desc;
  int nvec = l.eigsys(0).size;
  parallel_io io(get_num_nodes());
  generic_communicator<> com(io, desc, sizeof(su3_vector)*l.nt, generic_communicator<>::read);
  
  FILE* f;
  f = io.fopen(name.c_str(), "rb");

  su3_vector* buf = (su3_vector*) com.comm_buffer;
  for(int v=0; v<nvec; ++v)
  {
    com.execute(f);
    for(int i=0; i<desc.sites_on_node; ++i)
    for(int t=0; t<l.nt; ++t) l.eigsys(t).evec[v]->data[i] = buf[t+l.nt*i];
  }

  if(get_node_rank() == io.io_nodes[0])
  {
    for(int t=0; t<l.nt; ++t) fread(l.eigsys(t).eval, sizeof(double_complex), nvec, f);
  }

  if(io.io_nodes[0] == 0)
  for(int t=0; t<l.nt; ++t) broadcast((char*)l.eigsys(t).eval, sizeof(double_complex)*nvec);
  else{ printf("assumtion violated ... exiting\n"); abort(); }

  io.fclose(f);
}


}
