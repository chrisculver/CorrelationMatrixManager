#include "DIAGS/correlator.h"
#include "UTILS/wick.h"

void Correlator::wick_contract()
{
	std::vector<Diagram> new_diags;
	for(const auto &c_e: c.terms)
		for(const auto &a_e: a.terms)
		{
			new_diags=wick_contract_elems(c_e, a_e);
       
			///push some diags into diags
			///not duplicating elements
		}

  for(auto &d: diags)
    d.coef = d.coef*c.coef*a.coef;
}
