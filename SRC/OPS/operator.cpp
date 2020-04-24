#include "OPS/operator.h"

using namespace std;

Operator adjoint(const Operator o)
{
	Operator a = o;
	for(auto &t: a.terms)
		t=adjoint(t);


	///Include the p/m coefficient due to (gamma_4 GAMMA^{\dagger} gamma_4)
	///also each meson swaps the quarks picking up a minus sign.
	auto a_mesons = a.terms[0].mesons;
	auto pm_coef = 1;
	for(const auto &m : a_mesons)
	{	
	  pm_coef *= -1;
		auto g = m.gamma;
		if( (g=="5") || (g=="1") || (g=="2") || (g=="3") || (g=="1 5") || (g=="2 5") || (g=="3 5") )
			pm_coef*=-1;
	}

	if(pm_coef==-1)
		for(auto &t : a.terms)
			t.coef*=pm_coef;


	return a;
}
