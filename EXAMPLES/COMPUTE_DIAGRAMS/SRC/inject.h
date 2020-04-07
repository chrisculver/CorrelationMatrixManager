// inject.h
// A. Alexandru
// March 2013

// Both inject and extract produce dest[x] = src[x+offset]
// the difference is that inject doest this for all points in the source
// and extract does it for all points in the destination.

#pragma once

#include "cpu_vector.h"

namespace qcd {

void extract(const su3_field& src, su3_field& dest, const position &offset);

template <typename T>
void extract(int nvec, T*const* src, const lattice_desc &ls, 
    T** dest, const lattice_desc &ld, const position &offset);

template <typename T>
void extract(const generic_vector<T>& src, generic_vector<T>& dest, const position& offset)
{
  extract(1, &src.data, *src.desc, &dest.data, *dest.desc, offset);
}

template <typename T>
void inject(int nvec, T*const* src, const lattice_desc &ls, 
    T** dest, const lattice_desc &ld, const position &offset);

template <typename T>
void inject(const generic_vector<T>& src, generic_vector<T>& dest, const position& offset)
{
  inject(1, &src.data, *src.desc, &dest.data, *dest.desc, offset);
}

} // qcd namespace
