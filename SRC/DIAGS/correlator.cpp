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
		///First consolidate the diagrams returned amongst themselves.
		
		///push some diags into diags
		///not duplicating elements
		for(const auto &d: new_diags)
		{
			auto equivalent_diags = d.all_cyclic_related_diagrams();
			int same_diagram=-1;
			bool found=false;
			for(size_t t=0; t<equivalent_diags.size(); ++t)
			{
				if(!found)
				{
					for(size_t e=0; e<diags.size(); ++e)
					{
						if( (equivalent_diags[t]==(diags[e].traces)) && !found)
						{
							found=true;
							diags[e].coef+=d.coef;
						}			
					}
				}
			}
			if(!found)
				diags.push_back(d);	

		}
	}

	for(auto it = diags.begin(); it != diags.end(); it++)
		if( (*it).coef == 0)
			diags.erase(it--);

	///Include the p/m coefficient of the annihilation meson
	auto a_mesons = a.terms[0].mesons;
	int pm_coef = pow(-1,a_mesons.size());
	for(const auto &m : a_mesons)
	{	
		auto g = m.gamma;
		if( (g=="5") || (g=="1") || (g=="2") || (g=="3") || (g=="1 5") || (g=="2 5") || (g=="3 5") )
			pm_coef*=-1;
	}

	if(pm_coef*=-1)
		for(auto &d : diags)
			d.coef*=pm_coef;
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


void Correlator::compute_time_average_correlators(int NT)
{
  corr_t.resize(NT);
	for(int dt=0; dt<NT; ++dt)
  {
    complex<double> time_avg(0.,0.);
    for(int t=0; t<NT; ++t)
    {
      for(const auto& d : diags)
      {
        complex<double> trace_product(1.,0.);
        for(const auto& tr : d.traces)
        {
          trace_product *= tr.numerical_value[dt][t];
        }///end traces
        time_avg += complex<double>(d.coef,0)*trace_product;
      }///end diags
    }///end t
    corr_t[dt] = time_avg/(double(NT));
  }///end dt
	
}
