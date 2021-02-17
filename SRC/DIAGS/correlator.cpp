#include "DIAGS/correlator.h"
#include "UTILS/wick.h"
#include "UTILS/string_utilities.h"
#include "DIAGS/quark_line.h"

#include "spdlog/spdlog.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <map>

using namespace std;
using Saved_Diagrams = map<string, map<string,complex<double>>>;

//Correlator<QuarkLine>;

template<> void Correlator<QuarkLine>::wick_contract()
{
	auto wick_logger = spdlog::get("wick");

	vector<Diagram<QuarkLine>> new_diags;


	for(const auto &c_e: c.terms)
	for(const auto &a_e: a.terms)
	{
		wick_logger->debug("Doing elemental_"+
												to_string(&c_e - &c.terms[0])+
												to_string(&a_e - &a.terms[0]));

		new_diags=wick_contract_elems<QuarkLine>(c_e, a_e);

		wick_logger->debug("Done with elemental, adding to diags");
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
							///try to keep the list small...
							if(diags[e].coef==0)
								diags.erase( diags.begin() + e );
							break;
						}
					}
				}
				else
					break;
			}
			if(!found)
				diags.push_back(d);

		}
		wick_logger->debug("done adding new_diags");
	}

	///double check for zero diags
	for(auto it = diags.begin(); it != diags.end(); it++)
		if( (*it).coef == 0)
			diags.erase(it--);
}

template <> void Correlator<QuarkLine>::load_wick_contractions(const std::string filename, const int i, const int j)
{
	ifstream in_file(filename);

	std::string line;
	while(getline(in_file, line))
	{
		vector<string> log_split = split(line,'R');
		vector<string> equals_split = split(log_split[1],'=');
		if(equals_split[0]==("esulting diags for c_"+to_string(i)+"."+to_string(j)))
		{
			vector<string> diag_text = split(equals_split[1],'+');
			for(auto &term : diag_text)
			{
				Diagram<QuarkLine> d;
				vector<string> trs_text = split(term,'[');
				d.coef = stoi(trs_text[0]);
				for(size_t i=1; i<trs_text.size(); ++i)
				{
					Trace<QuarkLine> t;
					vector<string> qls_text = split(trs_text[i],'|');
					for(size_t j=0; j<qls_text.size(); ++j)
					{
						QuarkLine q;

						vector<string> disp_split = split(qls_text[j],'\\');
						q.gamma=disp_split[0].substr(1,disp_split[0].size()-2);
						vector<string> rhs = split(disp_split[1],' ');
						q.displacement = "\\" + rhs[0];
						q.mom=rhs[1]+" "+rhs[2]+" "+rhs[3];
						q.ti=rhs[4][0];
						q.tf=rhs[5][0];
						t.qls.push_back(q);
					}
					d.traces.push_back(t);
				}
				diags.push_back(d);
			}

		}
		line="";
	}
	in_file.close();
}

template <> void Correlator<QuarkLine>::load_numerical_results(Saved_Diagrams computed)
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
				Trace<QuarkLine> r = t;
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
			for(const auto &ti : ts)
				for(const auto &dt : dts)
					if( t.numerical_value.count(to_string(dt)+" "+to_string(ti)) == 0 )
						throw 't';
		}///end of trace loop
  }///end of diagram loop
}



template <> void Correlator<QuarkLine>::compute_time_average_correlators()
{
  corr_t.resize(dts.size());
	for(int i=0; i<dts.size(); ++i)
  {
		int dt = dts[i];
    complex<double> time_avg(0.,0.);
    for(const auto &t : ts)
    {
      for(const auto& d : diags)
      {
        complex<double> trace_product(1.,0.);
        for(auto tr : d.traces)
        {
          trace_product *= tr.numerical_value[to_string(dt)+" "+to_string(t)];
        }///end traces
        time_avg += complex<double>(d.coef,0)*trace_product;
      }///end diags
    }///end t
    corr_t[i] = time_avg/(double(ts.size()));
  }///end dt

}
