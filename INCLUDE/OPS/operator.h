#ifndef OPS_OPERATOR_H
#define OPS_OPERATOR_H

#include "OPS/elemental_op.h"

#include <vector>
#include <string> 


struct Operator
{
	public:
		std::vector<ElementalOp> terms;
	
		Operator(std::vector<ElementalOp> t):terms(t){};

		template<typename OStream>
		friend OStream &operator<<(OStream &os, const Operator &o)
		{
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
