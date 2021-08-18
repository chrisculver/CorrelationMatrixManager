#ifndef OPS_ELEMENTAL_OP_H
#define OPS_ELEMENTAL_OP_H

#include "meson.h"

#include <vector>

struct ElementalOp
{
	public:
		double coef;
		std::vector<Meson> mesons;

		ElementalOp(int c, std::vector<Meson> n_m):mesons(n_m),coef(c){};

		template<typename OStream>
		friend OStream &operator<<(OStream &os, const ElementalOp &e)
		{
			os << e.coef << "|";
			for(size_t i=0; i<e.mesons.size(); ++i)
			{
				os << e.mesons[i];
				if(i!=(e.mesons.size()-1))
					os << "|";
			}
			return os;
		}

};
		
ElementalOp adjoint(ElementalOp e);

#endif
