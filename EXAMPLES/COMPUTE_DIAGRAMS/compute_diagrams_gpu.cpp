/*
  This code computes the diagrams for meson spectroscopy by precomputing the
  quark lines.  The code here has two sources of input, one of which is a
  cpp file requiring recompilation of the program.  From ModernComputeCorrelator
  define_diagrams.cpp, run.in and diagram_names.txt are required.

unique_displacement:3 td 1 3
  define_diagrams.cpp has the code to multiply ql's into specific diagrams, i.e.
    M^{-1}(t,t_f)\Gamma(p)M^{-1}(t_f,t)\Gamma(p_2).

  WARNING : IT PRECOMPUTES QL FOR ALL COMBINATIONS OF MOM/GAMMA/DISP,
            extremely inefficient if those operators won't be used.
            This is where precomputing source/sink functions is good.

  The other file lists the unique_mom, displacement operators, gamma_matrices,
  and lattice information.

  The convention for quark line indices is
  Gamma_matrix, displacement, momenta.

  If this structure is changed ModernComputeCorrelator will need to be updated
  to reflect this correctly.
*/

#include "define_diagrams.h"

///Local Code by Culver
#include "SRC/gammas.h"
#include "SRC/gpu_kernel.h"
#include "SRC/io.h"
#include "SRC/mom_matrix.h"

///Local Code by Guo
#include "SRC/laplacean_eigensolver.h"
#include "SRC/smeared_matrix_object.h"
#include "SRC/lapmult.h"
#include "SRC/inject.h"
#include "SRC/gamma_mult.h"

///GWU-QCD Libraries
#include "comm/comm_low.h"
#include "options.h"
#include "timer.h"
#include "layout_minsurface.h"
#include "qcd_fields.h"
#include "smearing.h"
#include "boundary_flip.h"

///C++ Libraries
#include <omp.h>
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>

using namespace qcd;
using namespace std;

typedef Eigen::MatrixXcd mat;

template<typename Out> void split(const std::string &s, char delim, Out result)
{
  std::stringstream ss(s);
  std::string item;
  while(getline(ss, item, delim))
  {
    *(result++) = item;
  }
}
std::vector<std::string> split(const std::string &s, char delim)
{
  std::vector<std::string> elems;
  split(s, delim, back_inserter(elems));
  return elems;
}


struct tensmat
{
  mat m;
  const int nv;

  tensmat(int _n): m(4*_n, 4*_n), nv(_n){};

  Eigen::Block<mat> part(int i, int j){ return m.block(i*nv, j*nv, nv, nv); }
};


void get_matrix(smeared_matrix_object& sm, int tr, int tc, tensmat& m)
{
  for(int rv=0; rv<m.nv; ++rv)
  for(int rs=0; rs<4; ++rs)
  for(int cv=0; cv<m.nv; ++cv)
  for(int cs=0; cs<4; ++cs)
  {
    m.m(rs*m.nv+rv, cs*m.nv+cv).real( sm(tc, cv, cs, tr, rv, rs).real );
    m.m(rs*m.nv+rv, cs*m.nv+cv).imag( sm(tc, cv, cs, tr, rv, rs).imag );
  }
}


///Only 1 displacement is implemented here.
void create_matp_tens(std::vector<mat> &matp, std::vector<int> lat_size, int nvec, laplacean_eigensolver lapeig,
                                int px, int py, int pz, int this_node)
{
  int nx=lat_size[0]; int ny=lat_size[1]; int nz=lat_size[2]; int nt=lat_size[3];

  for(int t=0; t<nt; ++t)
  {
    for(int c=0; c<nvec; ++c)
    {
      colorvector tmp(&lapeig.desc);
      tmp = *lapeig.eigsys(t).evec[c];

#pragma omp parallel for default(none) shared(lapeig, tmp, nx, ny, nz, nt, px, py, pz, this_node), schedule(static)
      for(int i=0; i<lapeig.desc.sites_on_node; ++i)
      {
        position p = lapeig.desc.get_position(i,this_node);
        double_complex phase = exp(double_complex(0,1)*2*M_PI*(p[0]*px/((double)nx) + p[1]*py/((double)ny) + p[2]*pz/((double)nz)));
        tmp.data[i] *= phase;
      }

      for(int r=0; r<nvec; ++r)
      {
        double_complex res = cscalar_product(*lapeig.eigsys(t).evec[r],tmp);
        matp[t](r,c).real(res.real); matp[t](r,c).imag(res.imag);
      }
    }
  }


}



int main(int argc, char **argv)
{
	init_machine(argc,argv);
	int this_node = get_node_rank();
  options opt;

	std::vector<textline> mom_str, disp_str, gamma_str;
  std::string dt_list, t_list;
  std::string mname, lname, latname="auto";
	int nx, ny, nz, nt;
	bool checkeig = false;
	int nvec = 0;
	int cfg, NDIAGS;

  opt.add("nx",nx);
  opt.add("ny",ny);
  opt.add("nz",nz);
  opt.add("nt",nt);
  opt.add("latname", latname);
  opt.add("checkeig", checkeig);
  opt.add("nvec", nvec);
	opt.add("cfg", cfg);
	opt.add("ndiags", NDIAGS);
	opt.add("unique_mom", mom_str);
  opt.add("unique_gammas", gamma_str);
  opt.add("unique_displacement", disp_str);
  opt.add("dt_list", dt_list);
  opt.add("t_list", t_list);
  opt.read_options(argc,argv);

	lname = latname + ".laplacean.eigensystem";
	mname = latname + ".smeared.prop";

	timer all(false);
	all.start("all");

	smeared_matrix_object m(1,1);
	read_mtilde(m, mname, latname);

	layout_minsurface_eo desc(nx, ny, nz, nt);
	su3_field links(&desc);
	read_kentucky_lattice(latname, links);
	nhyp_smear_simple(links, links);
	int bc[4] = { 1, 1, 1, -1 };
	apply_boundary(links, bc);

	laplacean_eigensolver lapeig(links);
	read_lap(lapeig, m, checkeig, nvec, lname);

  std::vector<int> lat_size{nx,ny,nz,nt};
  ///Standard gwu-lqcd spectroscopy setup done.
  ///Convert mom,gamma,disp input
	std::vector< std::vector< int > > unique_mom(mom_str.size()),
                                    unique_gamma_names(gamma_str.size()),
                                    unique_displacements(disp_str.size());
  std::vector<mat> unique_gammas(gamma_str.size(),mat::Identity(4,4));///initialize to the id
	std::vector<mat> gamma(7);
	std::complex<double> I(0,1);
	///anti-milc basis
	gamma[1].resize(4,4); gamma[1] << 0, 0, 0, -I, 0, 0, -I, 0, 0, I, 0, 0, I, 0, 0, 0;
  gamma[2].resize(4,4); gamma[2] << 0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 0;
  gamma[3].resize(4,4); gamma[3] << 0, 0, -I, 0, 0, 0, 0, I, I, 0, 0, 0, 0, -I, 0, 0;
  gamma[4].resize(4,4); gamma[4] << 0, 0, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, 0, 0;
	gamma[1].resize(4,4);
  gamma[5] = gamma[1]*gamma[2]*gamma[3]*gamma[4];
  gamma[6] = mat::Identity(4,4);


	for(size_t i=0; i<mom_str.size(); ++i)
  {
		unique_mom[i].resize(3);
		std::stringstream ss(mom_str[i]);
		ss >> unique_mom[i][0] >> unique_mom[i][1] >> unique_mom[i][2];
	}
  for(size_t i=0; i<gamma_str.size(); ++i)
  {
    std::stringstream ss(gamma_str[i]);
    std::string tmp(ss.str());
		std::vector<std::string> ints = split(tmp,' ');
		for(size_t j=0; j<ints.size(); ++j)
		{
			unique_gammas[i]*=gamma[stoi(ints[j])];
		}
	}
  ///Take the gamma_names list and make the actual matrices
  //for(size_t gi=0; gi<unique_gammas[g].size(); ++gi)
  //  unique_gammas[gi]*=gamma[gi];

  ///TODO-Displacements : Actually implement something
  for(size_t i=0; i<disp_str.size(); i++)
  {
    unique_displacements[0]=std::vector<int>{0};
  }

  std::vector<int> unique_dts, unique_ts;
  std::cout << dt_list << "    " << t_list << std::endl;
  if(dt_list=="ALL")
  {
    for(size_t i=0; i<nt; ++i)
      unique_dts.push_back(i);
  }
  else if(dt_list=="HALF")/// up to and including half temporal extent of lattice
  {
    for(size_t i=0; i<=nt/2; ++i)
      unique_dts.push_back(i);
  }
  else
  {
    std::vector<std::string> times = split(dt_list,',');
    for(size_t i=0; i<times.size(); ++i)
      unique_dts.push_back(stoi(times[i]));
  }

  if(t_list=="ALL")
  {
    for(size_t i=0; i<nt; ++i)
      unique_ts.push_back(i);
  }
  else if(t_list=="HALF")/// exactly half
  {
    for(size_t i=0; i<nt/2; ++i)
      unique_ts.push_back(i);
  }
  else
  {
    std::vector<std::string> times = split(t_list,',');
    for(size_t i=0; i<times.size(); ++i)
      unique_ts.push_back(stoi(times[i]));
  }



  ///Compute the momentum matrices TODO-Displacements: For all mom and displacements
  ///How to select which ones are in operators?
  ///Currently does no displacement for all unique_mom;
  timer tm;
	tm.start("Momentum matrices");
  std::vector< std::vector<mat> > matp;

  for(size_t p=0; p<unique_mom.size(); ++p){
    std::vector<mat> mat(nt);
    for(int t=0; t<nt; ++t)
      mat[t].resize(nvec,nvec);

    printf("Computing mom matrix for p=(%d,%d,%d)\n",
        unique_mom[p][0], unique_mom[p][1], unique_mom[p][2]);

		create_matp_tens(mat, lat_size, nvec, lapeig,
      unique_mom[p][0], unique_mom[p][1], unique_mom[p][2],
                  this_node);
		matp.push_back(mat);
	}
	tm.stop();

  ///Container to hold all the diagrams
	std::vector < std::vector< std::vector< std::complex<double> > > > diag(NDIAGS);
	for(size_t i=0; i<diag.size(); ++i)
		diag[i].resize(unique_dts.size());
	for(size_t i=0; i<diag.size(); ++i)
		for(size_t j=0; j<diag[i].size(); ++j)
			diag[i][j].resize(unique_ts.size());
	for(size_t i=0; i<diag.size(); ++i)
		for(size_t j=0; j<diag[i].size(); ++j)
			for(size_t k=0; k<diag[i][j].size(); ++k)
				diag[i][j][k]=0.;

  for(int i_dt=0; i_dt<unique_dts.size(); ++i_dt)
  {
    int dt = unique_dts[i_dt];
    timer one_dt;
    one_dt.start("Computing qlines and diags for one dt");

    ///  Save q lines as 400 by 400 matrices.  for each momentum p, gamma, disp.
    for(int i_t=0; i_t<unique_ts.size(); ++i_t)
    {
      int t=unique_ts[i_t];
			timer tm_ql;
			tm_ql.start("computing qlines");

      int ngamma=unique_gammas.size();
      int ndisp=unique_displacements.size();
      int nmom=unique_mom.size();
			int NQL = ngamma*ndisp*nmom;
      std::vector< mat > ql_forward(NQL,mat::Zero(4*nvec,4*nvec)), ql_backward(NQL,mat::Zero(4*nvec,4*nvec)), ql_init(NQL,mat::Zero(4*nvec,4*nvec)), ql_fin(NQL,mat::Zero(4*nvec,4*nvec));

      for(size_t g=0; g<unique_gammas.size(); ++g)
      {
        mat gam = unique_gammas[g];

        for(size_t d=0; d<unique_displacements.size(); ++d)
        {
          for(size_t p=0; p<unique_mom.size(); ++p)
          {
            int tf = (t+dt)%nt;
            ///compute the ql
            tensmat tmp_init(nvec), tmp_fin(nvec);//, tmp_init(nvec), tmp_fin(nvec);
            tmp_init.m = mat::Zero(4*nvec, 4*nvec);
            tmp_fin.m = tmp_init.m;

            for(int s3=0; s3<4; ++s3)
            for(int s2=0; s2<4; ++s2)
            {
							if(abs(gam(s2,s3))!=0.)
              {
								//std::complex<double> tmp_g(gamma(s2,s3).real, gamma(s2,s3).imag);
                ///TODO-Displacements: Don't just use matp
//                if(gam == gamma[5])
//                  tmp_init.part(s2,s3) -= gam(s2,s3)*matp[p][t];
//                else
                tmp_init.part(s2,s3) += gam(s2,s3)*matp[p][t];
                tmp_fin.part(s2,s3) += gam(s2,s3)*matp[p][tf];
              }
            }///end of spin loops

/*            tensmat mpart(nvec);
            get_matrix(m, tf, t, mpart);
            ql_forward[g*ndisp*nmom + d*nmom + p] = mpart.m*tmp_init.m;
            get_matrix(m, t, tf, mpart);
            ql_backward[g*ndisp*nmom + d*nmom + p] = mpart.m*tmp_fin.m;
            ///TODO CHECK MATH IS CORRECT
            get_matrix(m, t, t, mpart);
            ql_init[g*ndisp*nmom + d*nmom + p] = mpart.m*tmp_init.m;
            get_matrix(m, tf, tf, mpart);
            ql_fin[g*ndisp*nmom + d*nmom + p] = mpart.m*tmp_fin.m;
*/
            tensmat mpart(nvec);
            get_matrix(m, t, tf, mpart);
            ql_forward[g*ndisp*nmom + d*nmom + p] = tmp_init.m*mpart.m;
            get_matrix(m, tf, t, mpart);
            ql_backward[g*ndisp*nmom + d*nmom + p] = tmp_fin.m*mpart.m;
            ///TODO CHECK MATH IS CORRECT
            get_matrix(m, t, t, mpart);
            ql_init[g*ndisp*nmom + d*nmom + p] = tmp_init.m*mpart.m;
            get_matrix(m, tf, tf, mpart);
            ql_fin[g*ndisp*nmom + d*nmom + p] = tmp_fin.m*mpart.m;
          }///end of mom
        }///end of displacement
      }///end of gammas
      tm_ql.stop();

      timer tm_diags;
      tm_diags.start("computing diagrams");
			define_diagrams(diag, ql_forward, ql_backward, ql_init, ql_fin, i_t, i_dt);
      tm_diags.stop();
		}/// end of t loop

    one_dt.stop();
	}/// end of dt loop

	all.stop();

  std::ifstream name_file("diagram_names.txt");
  std::string str;
  std::vector<std::string> diag_names;
  while(std::getline(name_file, str)){
    if(str.size()>0)
      diag_names.push_back(str);
  }
  name_file.close();

	std::ofstream corr_file;
	std::string cfg_str;
	std::stringstream ss;
	ss << std::setfill('0') << std::setw(3) << to_string(cfg);
	cfg_str  = ss.str();
	corr_file.open("diags_"+to_string(nx)+to_string(ny)+to_string(nz)+to_string(nt)+"_"+cfg_str+".dat");
	for(int i=0; i<NDIAGS; ++i){
		corr_file << diag_names[i] << "\n";
		for(int i_dt=0; i_dt<unique_dts.size(); ++i_dt){
			for(int i_t=0; i_t<unique_ts.size(); ++i_t){
        corr_file << std::fixed << std::setprecision(0) << unique_dts[i_dt] << " " << unique_ts[i_t] << " ";
				corr_file << std::scientific << std::setprecision(10) << diag[i][i_dt][i_t].real() << " " << diag[i][i_dt][i_t].imag() << "\n";
			}
		}
	}
  corr_file.close();


	if(this_node==0)
		printf("Total time: %0.2f s\n", all.get_time());

	shutdown_machine();

  return 0;
}
