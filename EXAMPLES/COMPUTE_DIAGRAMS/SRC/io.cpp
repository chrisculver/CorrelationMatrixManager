#include "io.h"

#include "boundary_flip.h"
#include "smearing.h"



using namespace qcd;  

void prep_links(su3_field &l, std::string latname)
{
// I tried making the desc here and then return l so that the main code read
// su3_field links( read_links(lat_size, latname) )
// but there seems to be a private copy constructor somewhere down the 
// class hierarchy. 
//
//  layout_minsurface_eo desc(lat_size[0],lat_size[1],lat_size[2],lat_size[3]);
//  su3_field l(&desc);

  read_kentucky_lattice(latname,l);
  nhyp_smear_simple(l,l);

  int bc[4]={1,1,1,-1};
  apply_boundary(l,bc);
}

void read_mtilde(smeared_matrix_object &m, std::string mname, std::string &latname)
{
  std::string newlatname;
  read(mname, m, newlatname);
  if(latname == "auto")
    latname = newlatname;

  printf("Lattice name: %s\n", latname.c_str());
  printf("nt: %02d nvec: %03d\n", m.nt, m.nvec);
}
void read_lap(laplacean_eigensolver &lap, smeared_matrix_object m, bool checkeig, int &nvec, std::string lname)
{
  lap.allocate_eigensystem(m.nvec);
  read_laplace_eigensystem(lname, lap);

  if(checkeig)
    lap.check_eigensystem();

  if(nvec==0)
    nvec = m.nvec;
}

void get_opts(int argc, char **argv, std::vector<int> &size, std::vector<std::string> &name, std::vector<textline> &mom1, bool &checkeig, int &nvec, int &cfg)
{
  options opt;

  opt.add("nx",size[0]);
  opt.add("ny",size[1]);
  opt.add("nz",size[2]);
  opt.add("nt",size[3]);
  opt.add("mtilde", name[0]);
  opt.add("lapeig", name[1]);
  opt.add("latname", name[2]);
  opt.add("mom", mom1);
  opt.add("checkeig", checkeig);
  opt.add("nvec", nvec);
	opt.add("cfg", cfg);

  opt.read_options(argc,argv);
}


