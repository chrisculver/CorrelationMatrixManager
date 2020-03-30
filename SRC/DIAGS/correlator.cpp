#include "DIAGS/correlator.h"
#include "UTILS/wick.h"

void Correlator::wick_contract()
{
	std::vector<Diagram> some_diags;
	for(const auto &c_e: c.terms)
		for(const auto &a_e: a.terms)
		{
			some_diags=wick_contract_elems(c_e, a_e);

			///push some diags into diags
			///not duplicating elements
		}

}
