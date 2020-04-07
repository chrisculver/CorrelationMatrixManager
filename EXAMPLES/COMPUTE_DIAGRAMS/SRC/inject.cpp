// inject.cpp
// A. Alexandru
// March 2013

// This routine copies the content of one lattice
// into another. The destination lattice is filled
// with the content of the source. When the destination
// is larger than the source, the source is tiled in
// a periodic fashion. The offset parameter indicates
// the position of the origin on the destination lattice
// with respect to the origin of the source lattice, i.e.
// dest(x) = src(x+offset)

#include "inject.h"
#include "comm/comm_intermediate.h"
#include "cpu_vector.h"
#include <vector>

namespace qcd {

template<class T>
static void execute_send(int nvec, T*const* src, T** dest, comm_list& cl, int stride = 1)
{
  offset_list_set os = create_offset_list_set(cl);
  linked_buffer_set lb;
  create_linked_buffer_set(lb, nvec, (char**)src, (char**)dest, os, sizeof(T), stride*sizeof(T));
  send_recv_buffer_set(lb);
  move_local_data(nvec, sizeof(T), (char**)src, (char**)dest, os);
}

template <typename T>
void extract(int nvec, T*const* src, const lattice_desc &ls, 
    T** dest, const lattice_desc &ld, const position &offset)
{
  std::vector<comm_entry> list;
  for(unsigned int i=0; i<ld.volume; ++i)
  {
    position p(i, ld);
    p = p + offset;
    p.normalize(ls);
    list.push_back(std::make_pair(p.index(ls), i));
  }

  comm_list cl(ls, ld, get_node_rank(), get_num_nodes(), list);
  execute_send(nvec, src, dest, cl);
}

template
void extract<wilson_vector>(int nvec, wilson_vector*const* src, const lattice_desc &ls, 
    wilson_vector** dest, const lattice_desc &ld, const position &offset);

template
void extract<su3_vector>(int nvec, su3_vector*const* src, const lattice_desc &ls, 
    su3_vector** dest, const lattice_desc &ld, const position &offset);

void extract(const su3_field& src, su3_field& dest, const position &offset)
{
  su3_matrix *psrc[4], *pdest[4];
  for(int dir=0; dir<4; ++dir)
  {
    psrc[dir]  = src.data  + dir*src.desc->sites_on_node;
    pdest[dir] = dest.data + dir*dest.desc->sites_on_node;
  }
  extract(4, psrc, *src.desc, pdest, *dest.desc, offset);
}

template <typename T>
void inject(int nvec, T*const* src, const lattice_desc &ls, 
    T** dest, const lattice_desc &ld, const position &offset)
{
  std::vector<comm_entry> list;
  for(unsigned int i=0; i<ls.volume; ++i)
  {
    position p(i, ls);
    p = p - offset;
    p.normalize(ld);
    list.push_back(std::make_pair(i,p.index(ld)));
  }

  comm_list cl(ls, ld, get_node_rank(), get_num_nodes(), list);
  execute_send(nvec, src, dest, cl);
}

template
void inject<wilson_vector>(int nvec, wilson_vector*const* src, const lattice_desc &ls, 
    wilson_vector** dest, const lattice_desc &ld, const position &offset);

template
void inject<su3_vector>(int nvec, su3_vector*const* src, const lattice_desc &ls, 
    su3_vector** dest, const lattice_desc &ld, const position &offset);

} // qcd namespace
