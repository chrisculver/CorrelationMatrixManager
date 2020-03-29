#include "OPS/operator.h"

using namespace std;

Operator adjoint(Operator o)
{
	for(auto &t: o.terms)
		t=adjoint(t);

	return o;
}
