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
};



#endif
