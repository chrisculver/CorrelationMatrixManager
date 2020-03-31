#ifndef OPS_OPERATOR_H
#define OPS_OPERATOR_H

#include "OPS/elemental_op.h"

#include <vector>
#include <string> 


struct Operator
{
	public:
		std::vector<ElementalOp> terms;
    int coef;

		Operator(int c, std::vector<ElementalOp> t):coef(c), terms(t){};

		template<typename OStream>
		friend OStream &operator<<(OStream &os, const Operator &o)
		{
      os << o.coef << "   ";
			for(size_t i=0; i<o.terms.size(); ++i)
			{
				os << o.terms[i];
				if(i!=(o.terms.size()-1))	
					os << " + ";
			}
			return os;
		}

};

Operator adjoint(Operator o);


#endif
