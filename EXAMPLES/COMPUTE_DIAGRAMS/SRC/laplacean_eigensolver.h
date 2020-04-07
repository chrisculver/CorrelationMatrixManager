// laplacean_eigensolver.h
// Guo Dehua
// Aug 2014

#pragma once

#include "colorvector.h"
#include "qcd_fields.h"
#include "vec_eigen_pair.h"
#include "layout_minsurface.h"


namespace qcd{

struct laplacean_eigensolver
{
  vec_eigen_pair<colorvector>& eigsys(int t) { return *eig[t]; }
  laplacean_eigensolver(su3_field& l) : eig(NULL), links(l), desc(l.desc->nx, l.desc->ny, l.desc->nz, 1), nt(l.desc->nt) {} ;
  void solve(int nvec, int order=0, double sigma=0);
  ~laplacean_eigensolver();
  void check_eigensystem();
  void allocate_eigensystem(int nvec);
private:
  vec_eigen_pair<colorvector>** eig;
  su3_field& links;
public:
  layout_minsurface desc;
  int nt;
private:
  void free_eigensystem();
};

void save_laplace_eigensystem(const std::string& name, laplacean_eigensolver& l);
void read_laplace_eigensystem(const std::string& name, laplacean_eigensolver& l);

}
