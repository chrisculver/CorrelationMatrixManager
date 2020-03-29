#include "OPS/elemental_op.h"

ElementalOp adjoint(ElementalOp e)
{
	for(auto &m: e.mesons)
		m=adjoint(m);

	return e;
}
