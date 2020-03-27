#ifndef OPS_ELEMENTAL_OP_H
#define OPS_ELEMENTAL_OP_H

#include "meson.h"

#include <vector>

struct ElementalOp
{
	public:
		int coef;
		std::vector<Meson> mesons;

		ElementalOp(int c, std::vector<Meson> n_m):mesons(n_m),coef(c){};
};
#endif
