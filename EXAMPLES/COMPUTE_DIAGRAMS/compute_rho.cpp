// compute_pion.cpp
// A. Alexandru
// Sep 2014

#include <vector>
#include <stdio.h>
#include "smeared_matrix_object.h"
#include "matrix.h"
#include "comm/comm_low.h"
#include "options.h"
#include "layout_minsurface.h"
#include "qcd_fields.h"
#include "laplacean_eigensolver.h"
#include "smearing.h"
#include "boundary_flip.h"
#include "timer.h"
#include "lapmult.h"
#include "gamma_mult.h"
#include "inject.h"
using namespace qcd;
double_complex compute_trace(smeared_matrix_object& m, std::vector<matrix>& matpi, std::vector<matrix>& matpf, int t, int tp, int s1, int s2, int s3, int s4, int dim)
{
  double_complex res = 0;
  matrix mproj_1(dim, dim), mproj_2(dim,dim), tmp1(dim, dim), tmp2(dim,dim);

  // note that <t,v,s|Mtilde|tp,vp,sp> = m(tp, vp, sp, t, v, s)
  for(int r=0; r<dim; ++r) for(int c=0; c<dim; ++c) { mproj_1(r,c) = m(t, c, s2, tp, r, s1); mproj_2(r,c) = m(t, c, s4, tp, r, s3);}
Eigen::setNbThreads(1);

  tmp1 = mproj_2*matpi[t];
  tmp2 = matpf[tp]*mproj_1;
  for(int i=0; i<dim*dim; ++i) res += conj(tmp2[i])*tmp1[i]; 

  return res;
}

double_complex rho1(int i, int r, int c, bool dag=false)
{
  static matrix g[3] = { gamma_matrix.g1(), gamma_matrix.g2(), gamma_matrix.g3()};
  
  // gamma4 gammai^\dag gamma4 = -gammai
  if(dag) return -g[i](r,c);
  return g[i](r,c);
}

double_complex rho2(int i, int r, int c, bool dag=false)
{
  static matrix g[3] = { gamma_matrix.g1(), gamma_matrix.g2(), gamma_matrix.g3()};
  matrix res(4,4);
  res = gamma_matrix.g4()*g[i];
  
  // gamma4 (gamma4 gammai)^\dag gamma4 = gamma4 gammai so we return the same thing
  if(dag) return res(r,c);
  return res(r,c);
}

double_complex rho3(int i, int r, int c, bool dag=false)
{
  static matrix g[3] = { gamma_matrix.g1(), gamma_matrix.g2(), gamma_matrix.g3()};
  
  // gamma4 gammai^\dag gamma4 = -gammai
  if(dag) return -g[i](r,c);
  return g[i](r,c);
}

double_complex rho4(int i, int r, int c, bool dag=false)
{
  matrix res(4,4);
  res = gamma_matrix.g4()*gamma_matrix.g4();
  if(dag) return -0.5*res(r,c);
  return 0.5*res(r,c);
}

int main(int argc, char** argv)
{
  init_machine(argc, argv);
  options opt;
  std::string mname, lname, latname="auto";
  std::vector<textline> mom;
  int nx, ny, nz, nt;
  bool checkeig = true;
  int nvec = 0;
  opt.add("nx", nx);
  opt.add("ny", ny);
  opt.add("nz", nz);
  opt.add("nt", nt);
  opt.add("mtilde", mname);
  opt.add("lapeig", lname);
  opt.add("latname", latname);
  opt.add("mom", mom);
  opt.add("checkeig", checkeig);
  opt.add("nvec", nvec);
  opt.read_options(argc, argv);
  int this_node = get_node_rank();

  timer all(false);
  all.start("all");

  smeared_matrix_object m(1, 1);
  std::string newlatname;
  read(mname, m, newlatname);
  if(latname == "auto") latname = newlatname;
  printf("lattice name: %s\n", latname.c_str());
  printf("nt: %02d nvec: %03d\n", m.nt, m.nvec);

  layout_minsurface_eo desc(nx, ny, nz, nt);
  su3_field links(&desc);
  read_kentucky_lattice(latname, links);
  nhyp_smear_simple(links, links);
  int bc[4] = {1,1,1,-1};
  apply_boundary(links, bc);

  laplacean_eigensolver lapeig(links);
  lapeig.allocate_eigensystem(m.nvec);
  read_laplace_eigensystem(lname, lapeig);
  if(checkeig) lapeig.check_eigensystem();

  if(nvec==0) nvec = m.nvec;
  std::vector<matrix> matp(nt);
  std::vector<matrix> matps(nt);
  std::vector<matrix>** matpc;
  matpc = new std::vector<matrix>*[3];
  for(int i=0; i<3; ++i) matpc[i] = new std::vector<matrix>(nt); 
  for(int t=0; t<nt; ++t)
  {  
    matp[t].resize(nvec, nvec);
    matps[t].resize(nvec, nvec);
    for(int i=0; i<3; ++i) (*matpc[i])[t].resize(nvec, nvec);
  }
  for(unsigned int p=0; p<mom.size(); ++p)
  {
    std::stringstream ss(mom[p]);
    int px, py, pz; ss >> px >> py >> pz;
    if(this_node == 0) printf("Propagator for momentum %d [%d %d %d]\n", p, px, py, pz);

    timer tm;
    tm.start("momentum matrices");
    // create the momentum matrices
    for(int t=0; t<nt; ++t) for(int c=0; c<nvec; ++c)
    {
      colorvector tmp(&lapeig.desc);
      tmp = *lapeig.eigsys(t).evec[c];
#pragma omp parallel for default(none) shared(lapeig, tmp, nx, ny, nz, nt, px, py, pz, this_node) schedule(static)
      for(unsigned int i=0; i<lapeig.desc.sites_on_node; ++i)
      {
	position p = lapeig.desc.get_position(i, this_node);
	double_complex phase = exp(double_complex(0,-1)*2*M_PI*(p[0]*px/((double)nx) + p[1]*py/((double)ny) + p[2]*pz/((double)nz)));
	tmp.data[i] *= phase;
      }

      for(int r=0; r<nvec; ++r) matp[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r], tmp); 
    }
    tm.stop();
    tm.start("momentum matrices smeared");
    // create the smeared momentum matrices
    {
      vec_eigen_pair<colorvector> *dl[3];
      for(int j=0; j<3; ++j) dl[j] = new vec_eigen_pair<colorvector>(nvec, lapeig.desc);
      su3_field linksproj(&lapeig.desc);
      lapmult lm(linksproj);
      for(int t=0; t<nt; ++t) 
      {
	extract(links, linksproj, position(0, 0, 0, t));

	for(int c=0; c<nvec; ++c) for(int r=0; r<nvec; ++r) matps[t](r,c) = 0;
	for(int c=0; c<nvec; ++c) for(int j=0; j<3; ++j) lm.covderiv(j, *lapeig.eigsys(t).evec[c], *dl[j]->evec[c]);

	for(int c=0; c<nvec; ++c)
	{
	  for(int j=0; j<3; ++j)
	  {
	    colorvector tmp(&lapeig.desc);
	    tmp = *dl[j]->evec[c];
#pragma omp parallel for default(none) shared(lapeig, tmp, nx, ny, nz, nt, px, py, pz, this_node) schedule(static)
	    for(unsigned int i=0; i<lapeig.desc.sites_on_node; ++i)
	    {
	      position p = lapeig.desc.get_position(i, this_node);
	      double_complex phase = exp(double_complex(0,-1)*2*M_PI*(p[0]*px/((double)nx) + p[1]*py/((double)ny) + p[2]*pz/((double)nz)));
	      tmp.data[i] *= phase;
	    }

	    for(int r=0; r<nvec; ++r) matps[t](r,c) += cscalar_product(*dl[j]->evec[r], tmp); 
	  }
	}
      }
      for(int j=0; j<3; ++j) delete dl[j];
    }
    tm.stop();

    tm.start("momentum matrices smeared commutator");
    // create the smeared anticomutator momentum matrices
    {
      vec_eigen_pair<colorvector> *dl[3];
      for(int j=0; j<3; ++j) dl[j] = new vec_eigen_pair<colorvector>(nvec, lapeig.desc);
      su3_field linksproj(&lapeig.desc);
      lapmult lm(linksproj);
      for(int t=0; t<nt; ++t) 
      {
	extract(links, linksproj, position(0, 0, 0, t));

	for(int i=0; i<3; ++i) for(int c=0; c<nvec; ++c) for(int r=0; r<nvec; ++r) (*matpc[i])[t](r,c) = 0;
	for(int c=0; c<nvec; ++c) for(int j=0; j<3; ++j) lm.covderiv(j, *lapeig.eigsys(t).evec[c], *dl[j]->evec[c]);

	for(int c=0; c<nvec; ++c)
	{
	  for(int j =0; j<3; ++j)
          {
            colorvector tmp(&lapeig.desc);
            colorvector tmp1(&lapeig.desc);
	    tmp  = *dl[j]->evec[c];
            tmp1= *lapeig.eigsys(t).evec[c];
#pragma omp parallel for default(none) shared(lapeig, tmp, tmp1, nx, ny, nz, nt, px, py, pz, this_node) schedule(static)
	    for(unsigned int i=0; i<lapeig.desc.sites_on_node; ++i)
	    {
	      position p = lapeig.desc.get_position(i, this_node);
	      double_complex phase = exp(double_complex(0,-1)*2*M_PI*(p[0]*px/((double)nx) + p[1]*py/((double)ny) + p[2]*pz/((double)nz)));
	      tmp.data[i] *= phase;
	      tmp1.data[i] *= phase;
	    }
          
	    for(int r=0; r<nvec; ++r) (*matpc[j])[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r], tmp)-cscalar_product(*dl[j]->evec[r], tmp1); 
	  }
        }
         
      }
      for(int j=0; j<3; ++j) delete dl[j];
    }
    tm.stop();

    matrix g5(4,4);
    g5 = gamma_matrix.g5();
    tm.start("propagator calculation");
    //std::vector<double_complex> prop(m.nt);
    std::vector<double_complex> **prop;
    int num = 3*16;
    prop = new std::vector<double_complex>*[num];
    for (int i = 0; i < num; ++i) prop[i] = new std::vector<double_complex>(m.nt); 

#pragma omp parallel for default(none) shared(m, prop, matp, matps, matpc, nvec, g5, num) schedule(static, 2)
    for(int dt=0; dt<m.nt; ++dt)
    {
      double_complex res[num]; for( int i = 0; i < num; ++i) res[i] = 0;
      for(int t=0; t<m.nt; ++t)
      for(int s1=0; s1<4; ++s1)
      for(int s2=0; s2<4; ++s2)
      for(int s3=0; s3<4; ++s3)
      for(int s4=0; s4<4; ++s4)
      { 
        // compute rho1_i to rho1_i
        for(int i=0; i<3; ++i)
        {
          double_complex coef = -rho1(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho1(i, s1, s3);
          if( abs(coef) !=0 ) res[i] += coef*compute_trace(m, matp, matp, t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute < rho1_i rho2_i^dagger >
        for(int i=0; i<3; ++i)
        {
          double_complex coef = -rho2(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho1(i, s1, s3);
          if( abs(coef) !=0 ) res[3+i] += coef*compute_trace(m, matp, matp, t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute < rho1_i rho3_i^dagger >
        for(int i=0; i<3; ++i)
        {
          double_complex coef = -rho3(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho1(i, s1, s3);
          if( abs(coef) !=0 ) res[6+i] += coef*compute_trace(m, matps, matp, t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute < rho1_i rho4_i^dagger >
        for(int i=0; i<3; ++i)
        {
          double_complex coef = -rho4(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho1(i, s1, s3);
          if( abs(coef) !=0 ) res[9+i] += coef*compute_trace(m, *matpc[i], matp, t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
 	// compute < rho2_i rho1_i^dagger >
        for(int i=0; i<3; ++i)
        {
          double_complex coef = -rho1(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho2(i, s1, s3);
          if( abs(coef) !=0 ) res[12+i] += coef*compute_trace(m, matp, matp, t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute rho2_i to rho2_i
        for(int i=0; i<3; ++i)
        {
          double_complex coef = -rho2(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho2(i, s1, s3);
          if( abs(coef) !=0 ) res[15+i] += coef*compute_trace(m, matp, matp, t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute <rho2_i  rho3_i^dagger>
        for(int i=0; i<3; ++i)
        {
          double_complex coef = -rho3(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho2(i, s1, s3);
          if( abs(coef) !=0 ) res[18+i] += coef*compute_trace(m, matps, matp, t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute <rho2_i  rho4_i^dagger>
        for(int i=0; i<3; ++i)
        {
          double_complex coef = -rho4(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho2(i, s1, s3);
          if( abs(coef) !=0 ) res[21+i] += coef*compute_trace(m, *matpc[i], matp, t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute < rho3_i rho1_i^dagger >
        for(int i=0; i<3; ++i)
        {
          double_complex coef = -rho1(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho3(i, s1, s3);
          if( abs(coef) !=0 ) res[24+i] += coef*compute_trace(m, matp, matps, t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute < rho3_i rho2_i^dagger >
        for(int i=0; i<3; ++i)
        {
          double_complex coef = -rho2(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho3(i, s1, s3);
          if( abs(coef) !=0 ) res[27+i] += coef*compute_trace(m, matp, matps, t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute rho3_i to rho3_i
         for(int i=0; i<3; ++i)
        {
          double_complex coef = -rho3(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho3(i, s1, s3);
          if( abs(coef) !=0 ) res[30+i] += coef*compute_trace(m, matps, matps, t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute <rho3_i  rho4_i^dagger >
         for(int i=0; i<3; ++i)
        {
          double_complex coef = -rho4(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho3(i, s1, s3);
          if( abs(coef) !=0 ) res[33+i] += coef*compute_trace(m, *matpc[i], matps, t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute <rho4_i  rho1_i^dagger > There is a minus sign in the coef because of the dagger of rho4 doing the optimization
         for(int i=0; i<3; ++i)
        {
          double_complex coef = rho1(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho4(i, s1, s3);
          if( abs(coef) !=0 ) res[36+i] += coef*compute_trace(m, matp, *matpc[i], t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute <rho4_i  rho2_i^dagger > There is a minus sign in the coef because of the dagger of rho4 doing the optimization
         for(int i=0; i<3; ++i)
        {
          double_complex coef = rho2(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho4(i, s1, s3);
          if( abs(coef) !=0 ) res[39+i] += coef*compute_trace(m, matp, *matpc[i], t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
        // compute <rho4_i  rho3_i^dagger >There is a minus sign in the coef because of the dagger of rho4 doing the optimization
         for(int i=0; i<3; ++i)
        {
          double_complex coef = rho3(i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rho4(i, s1, s3);
          if( abs(coef) !=0 ) res[42+i] += coef*compute_trace(m, matps, *matpc[i], t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        } 
        // compute rho4_i to rho4_i  There is a minus sign in the coef because of the dagger of rho4 doing the optimization
        for(int i=0; i<3; ++i)
        { 
          double_complex coef = rho4(i, s4, s2, true)*g5(s2,s2)*g5(s1,s1)*rho4(i, s1, s3);
          if( abs(coef) !=0 ) res[45+i] += coef*compute_trace(m, *matpc[i], *matpc[i], t, (t+dt)%m.nt, s1, s2, s3, s4, nvec);
        }
      }
        for( int i =0; i < num; ++i) (*prop[i])[dt] = res[i]/m.nt;
        //  prop[dt]=res/m.nt;
    }
    tm.stop();
    
    for(int j=0; j<num/3; ++j)
    for(int i=0; i<m.nt; ++i)
    {
      printf("MOM%01d%01d%01d RHO%02d Gt= %02d ", px, py, pz, j, i);
      for(int d=0; d<3; ++d)
      printf("%+20.15e %+20.15e ",(*prop[3*j+d])[i].real,(*prop[3*j+d])[i].imag);
      printf("\n");
    }
    for (int i=0; i<num; ++i) delete prop[i];
    delete [] prop; 
  }

  all.stop();
  if(this_node == 0) printf("Total time: %.2f s\n", all.get_time());
  shutdown_machine();
  return 0;
}
