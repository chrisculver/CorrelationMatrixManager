#include "gammas.h"

using namespace qcd;

double_complex gammaid(int r, int c, bool dag)
{
	if(r==c)
		return double_complex(1,0);
	else
		return double_complex(0,0);
}

double_complex gamma5(int r, int c, bool dag)
{
	matrix g5 = gamma_matrix.g5();
	if(dag)
		return -g5(r,c);
	else
		return g5(r,c);	
}

double_complex gammai(int i, int r, int c, bool dag)
{
	static matrix g[3] = { gamma_matrix.g1(), gamma_matrix.g2(), gamma_matrix.g3() };
	if(dag)
		return -g[i](r,c);
	else
		return g[i](r,c);
}

double_complex gammaigamma5(int i, int r, int c, bool dag)
{
	static matrix gi[3] = { gamma_matrix.g1(), gamma_matrix.g2(), gamma_matrix.g3() };
	matrix g(4,4);

	g = gi[i]*gamma_matrix.g5();

	if(dag)
		return -g(r,c);
	else
		return g(r,c);	
}
