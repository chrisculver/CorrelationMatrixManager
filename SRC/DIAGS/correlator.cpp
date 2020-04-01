#include "DIAGS/correlator.h"
#include "UTILS/wick.h"

#include <algorithm>

using namespace std;
using Saved_Traces = map<string, vector<vector<complex<double>>> >;

void Correlator::wick_contract()
{
	vector<Diagram> new_diags;
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


void Correlator::load_numerical_results(Saved_Traces computed)
{
	for(auto& d : diags)
  {
		for(auto& t : d.traces)
    {
      if( computed.count(t.name()) > 0 )
        t.numerical_value = computed[t.name()];
      else
      {
				///Search for cyclic permutations of the trace.
				Trace r = t;
				bool found = false;

				///TODO is this one extra permutation then necessary?
				for(size_t i=0; i<r.qls.size(); ++i)
				{
					rotate(r.qls.begin(), r.qls.begin()+1, r.qls.end());
					if( computed.count(r.name()) > 0 )
					{
						t.numerical_value = computed[r.name()];
						found=true;
						break;
					}
				}
        
       

				if(!found)
          throw 'm';
      }
    }
  }
}
