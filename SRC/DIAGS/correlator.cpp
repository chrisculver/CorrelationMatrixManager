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
		for(const auto &d: new_diags)
		{
			auto equivalent_traces = d.all_related_traces();
			int same_diagram=-1;
			bool found=false;
			for(size_t t=0; t<equivalent_traces.size(); ++t)
			for(size_t e=0; e<diags.size(); ++e)
			{
				if(equivalent_traces[t]==(diags[e].traces))
				{
					found=true;
					diags[e].coef+=d.coef;
					break;
				}			
			}
			if(!found)
				diags.push_back(d);		

		}
	}

  for(auto &d: diags)
    d.coef = d.coef*c.coef*a.coef;
}
