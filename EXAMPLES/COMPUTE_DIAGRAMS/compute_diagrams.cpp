// compute_scalarisosinglet_zeromom.cpp
// A. Alexandru
// July 2015

#include <vector>
#include <set>
#include <map>
#include <stdio.h>
#include "SRC/smeared_matrix_object.h"
#include "comm/comm_low.h"
#include "options.h"
#include "layout_minsurface.h"
#include "qcd_fields.h"
#include "smearing.h"
#include "SRC/laplacean_eigensolver.h"
#include "boundary_flip.h"
#include <Eigen/Dense>

using namespace qcd;

// For the scalar isosinglet (ubar u + dbar d)/\sqrt2
// in the isospin limit m_u = m_d
// there are two contributions to the correlator:
// a connected one -1*Tr S_(0,t) S(t,0)
// ans a disconnected one +2*Tr S_(0,0) Tr S_(t,t)

struct step
{
  int gamma;
  std::vector<int> p;
  int t;
};

typedef Eigen::MatrixXcd mat;

struct tensmat
{
  mat m;
  //const int nv;
  int nv;

  tensmat(int _n): m(4*_n, 4*_n), nv(_n){};

  Eigen::Block<mat>
  part(int i, int j) { return m.block(i*nv, j*nv, nv, nv); }

};

void compute_momentum_matrix(laplacean_eigensolver& l, int t, const std::vector<int>& mom, mat& mp)
{
  double_complex I(0,1);
  rank_t this_node = get_node_rank();
  colorvector expvec(&l.desc);
  for(uint i=0; i<l.desc.sites_on_node; ++i)
  {
    position p = l.desc.get_position(i, this_node);
    for(int c=0; c<3; ++c)
    expvec.data[i].c[c] = exp(I*(mom[0]*p[0]*2*M_PI/l.desc.nx+mom[1]*p[1]*2*M_PI/l.desc.ny+mom[2]*p[2]*2*M_PI/l.desc.nz));
  }

  vec_eigen_pair<colorvector>& lapvec = l.eigsys(t);

  for(uint i=0; i<lapvec.size; ++i)
  {
    colorvector tmp(&l.desc);
    for(uint k=0; k<l.desc.sites_on_node; ++k)
    for(int c=0; c<3; ++c) tmp.data[k].c[c] = expvec.data[k].c[c]*(lapvec.evec[i]->data[k].c[c]);
    for(uint j=0; j<lapvec.size; ++j) 
    {
      double_complex res = cscalar_product(*lapvec.evec[j], tmp);
      mp(j,i).real(res.real); mp(j,i).imag(res.imag);
    }
  }
    
}

void get_matrix(smeared_matrix_object& sm, int tr, int tc, tensmat& m)
{
  for(int rv=0; rv<m.nv; ++rv)
  for(int rs=0; rs<4; ++rs)
  for(int cv=0; cv<m.nv; ++cv)
  for(int cs=0; cs<4; ++cs)
  {
    m.m(rs*m.nv+rv, cs*m.nv+cv).real(sm(tc, cv, cs, tr, rv, rs).real);
    m.m(rs*m.nv+rv, cs*m.nv+cv).imag(sm(tc, cv, cs, tr, rv, rs).imag);
  }
}

int main(int argc, char** argv)
{
  init_machine( argc, argv);
  int nx, ny, nz, nt, nvec;
  std::string latname, lap_name, mat_name;
  std::vector<textline> diagrams;
  options opt;
  opt.add("nx", nx);
  opt.add("ny", ny);
  opt.add("nz", nz);
  opt.add("nt", nt);
  opt.add("nvec",nvec);
  opt.add("latname", latname);
  opt.add("diagrams", diagrams);
  opt.read_options(argc, argv);
  
  std::vector<std::vector<step> > allsteps;
  for(int i=0; i<diagrams.size(); ++i)
  { 
    textline &diagram = diagrams[i];
    std::vector<step> steps;
    std::stringstream ss(diagram);
    step tmp; tmp.p.resize(3);
    while(ss >> tmp.gamma >> tmp.p[0] >> tmp.p[1] >> tmp.p[2] >> tmp.t) steps.push_back(tmp);
    allsteps.push_back(steps);
  }

  lap_name = latname+".laplacean.eigensystem";
  mat_name = latname+".smeared.prop";

  layout_minsurface_eo l( nx, ny, nz, nt);
  layout_minsurface lp( nx, ny, nz, 1);

  su3_field links(&l);
  read_kentucky_lattice(latname, links);
  nhyp_smear_simple(links, links);
  int bc[4] ={1,1,1,-1};
  apply_boundary(links, bc);
  laplacean_eigensolver lapeig(links);
 
  lapeig.allocate_eigensystem(nvec);
  read_laplace_eigensystem(lap_name, lapeig);
  //lapeig.check_eigensystem(); 

  smeared_matrix_object m(1, 1);
  read(mat_name, m, latname);

  printf("lattice name: %s\n", latname.c_str());
  printf("nt: %02d nvec: %03d\n", m.nt, m.nvec);

  // precompute the momentum matrices
  std::set<std::vector<int> > allmom; for(int i=0; i<allsteps.size(); ++i) {std::vector<step> &steps = allsteps[i]; for(uint i=0; i<steps.size(); ++i) allmom.insert(steps[i].p);}
  std::vector<std::map<std::vector<int>, mat*> > momentum(nt);
  std::vector<mat> _momentum(nt*allmom.size());
  int k=0;
  for(std::set<std::vector<int> >::iterator it = allmom.begin(); it != allmom.end(); ++it) 
  {
    printf("p: %d %d %d\n", (*it)[0], (*it)[1], (*it)[2]); 
    for(int t=0; t<nt; ++t)
    {
      momentum[t][*it] = &_momentum[k++];
      momentum[t][*it]->resize(nvec, nvec);
      compute_momentum_matrix(lapeig, t, *it, *momentum[t][*it]);
    }
  }


  std::complex<double> I(0,1);
  std::vector<mat> gamma(16); gamma[0] = mat::Identity(4,4);
#if 0
  // pauli sakurai baisi
  gamma[1].resize(4,4); gamma[1] << 0, 0, 0, -I, 0, 0, -I, 0, 0, I, 0, 0, I, 0, 0, 0;
  gamma[2].resize(4,4); gamma[2] << 0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0;
  gamma[3].resize(4,4); gamma[3] << 0, 0, -I, 0, 0, 0, 0, I, I, 0, 0, 0, 0, -I, 0, 0;
  gamma[4].resize(4,4); gamma[4] << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1;
#endif
  // anti-milc basis
  //gamma[1].resize(4,4); gamma[1] << 0, 0, 0, -I, 0, 0, -I, 0, 0, I, 0, 0, I, 0, 0, 0;
  //gamma[2].resize(4,4); gamma[2] << 0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 0;
  //gamma[3].resize(4,4); gamma[3] << 0, 0, -I, 0, 0, 0, 0, I, I, 0, 0, 0, 0, -I, 0, 0;
  //gamma[4].resize(4,4); gamma[4] << 0, 0, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, 0, 0;
  // euclidean gammas
  gamma[1].resize(4,4); gamma[1] << 0, 0, 0, I, 0, 0, I, 0, 0, -I, 0, 0, -I, 0, 0, 0;
  gamma[2].resize(4,4); gamma[2] << 0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 0;
  gamma[3].resize(4,4); gamma[3] << 0, 0, I, 0, 0, 0, 0, -I, -I, 0, 0, 0, 0, I, 0, 0;
  gamma[4].resize(4,4); gamma[4] << 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0;  
  gamma[5] = gamma[1]*gamma[2]*gamma[3]*gamma[4];
  gamma[6] = mat::Identity(4,4);
  gamma[11] = gamma[1]*gamma[5];
	gamma[12] = gamma[2]*gamma[5];
	gamma[13] = gamma[3]*gamma[5];	



  // assume that there are only two times: i(0) f(1)
  // assume that there are 6 steps with times alternating 0, 1, 0, 1, 0, 1
  std::vector<std::vector<std::complex<double> > > final_res(allsteps.size());
  for(int i=0; i<allsteps.size(); ++i) final_res[i] = std::vector<std::complex<double> >(nt);
  for(int i=0; i<allsteps.size(); ++i)
  {

  std::vector<step> steps=allsteps[i]; 
  printf("[ ");
  for(uint i=0; i<steps.size(); ++i) 
  {
    step& s = steps[i];
    if(s.gamma==5)
      printf("5 ");
    else if(s.gamma==6)
      printf("6 ");
    else if(s.gamma==1)
      printf("1 ");
		else if(s.gamma==11)
			printf("1 5 ");
		printf("\\delta_{ii} ");
    printf("%d %d %d ", s.p[0], s.p[1], s.p[2]);
    if(s.t==0)
      printf("i ");
    else if(s.t==1)
      printf("f ");
		if(steps[(i+1)%steps.size()].t==0)
			printf("i ");
		else
			printf("f ");
    if(i<steps.size()-1) printf("| "); else printf("]\n");
  }
  for(int dt=0; dt<m.nt; ++dt)
  {
  for(int t=0; t<m.nt; ++t)
  {
    int tf = (t+dt)%m.nt;
    int ts[2] = {t,tf};
    tensmat mp(nvec), res(nvec), tmp(nvec);
    res.m = mat::Identity(4*nvec, 4*nvec);
   
    for(uint k=0; k<steps.size(); ++k)
    {
      get_matrix(m, ts[steps[k].t], ts[steps[(k+1)%steps.size()].t], mp);
      tmp.m = mat::Zero(4*nvec,4*nvec);
      for(int rs=0; rs<4; ++rs)
      for(int cs=0; cs<4; ++cs)
      for(int is=0; is<4; ++is) if(abs(gamma[steps[k].gamma](is,cs))!=0.0)
      {
        tmp.part(rs,cs)+=gamma[steps[k].gamma](is,cs)*res.part(rs,is)*(*(momentum[ts[steps[k].t]][steps[k].p]));
      }
      res.m = tmp.m*mp.m;
    }

      final_res[i][dt] += res.m.trace();
       
      printf("%7.10e",res.m.trace().real());
      printf(" %7.10e ",res.m.trace().imag());
         
//if(t==nt-1)printf("\nres d: %03d dt: %02d res: % 17.10e % 17.10e\n\n", i, dt, final_res[i][dt].real()/nt, final_res[i][dt].imag()/nt);
fflush(stdout);
  }
  printf("\n");
  }
  }

  shutdown_machine();
  return 0;
}
