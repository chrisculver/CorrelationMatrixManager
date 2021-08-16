#include "DIAGS/correlator.h"
#include "UTILS/wick.h"
#include "UTILS/string_utilities.h"
#include "DIAGS/quark_line.h"

#include "spdlog/spdlog.h"

#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>
#include <unordered_map>

using namespace std;
using Saved_Diagrams = map<string, vector<vector<complex<double>>>>;

template<> void Correlator<QuarkLine>::wick_contract()
{
	auto wick_logger = spdlog::get("wick");

	set<Diagram<int>> all_short_diags;
	int ql_idx=0;
	map<int, QuarkLine> int_to_QL;
	map<QuarkLine, int> QL_to_int;

	for(const auto &c_e: c.terms)
	for(const auto &a_e: a.terms)
	{
//		wick_logger->debug("Doing elemental_"+
//												to_string(&c_e - &c.terms[0])+
//												to_string(&a_e - &a.terms[0]));

		auto new_diags=wick_contract_elems<QuarkLine>(c_e, a_e);
		std::vector<Diagram<int>> short_diags;
		for(const auto &d: new_diags)
		{
			std::vector<Trace<int>> short_vt;
			for(const auto &t: d.traces)
			{
				std::vector<int> short_vql;
				for(const auto &ql: t.qls)
				{
					//cout << "diagram " << to_string(&d - &new_diags[0]) << endl;
					//cout << ql.gamma << ql.displacement << ql.ti << ql.tf << ql.mom << endl;
					//cout << (ql.gamma < int_to_QL[0].gamma) << endl;
					//cout << (ql < int_to_QL[0]) << endl;
					//cout << (QL_to_int.find(ql)!=QL_to_int.end()) << endl;
					if(QL_to_int.find(ql)!=QL_to_int.end())
					{
						short_vql.push_back( QL_to_int[ql] );
					}
					else
					{
						QL_to_int.insert( std::pair<QuarkLine, int>(ql, ql_idx) );
						int_to_QL.insert( std::pair<int, QuarkLine>(ql_idx, ql) );

						short_vql.push_back(ql_idx);
						ql_idx++;
					}
				}

				Trace<int> tshort;
				tshort.qls = short_vql;
				short_vt.push_back( tshort );
			}
			short_diags.push_back(Diagram<int>(d.coef,short_vt));
		}

		for(auto &d : short_diags)
		{
//			auto all_perms = d.all_cyclic_related_trace_products();
//			sort(all_perms.begin(), all_perms.end());

//			d.traces = all_perms[0];

//			auto tst = d;
			d.order_traces();
		}

//		wick_logger->debug("Done with elemental, adding {:d} short_diags to all_short_diags", short_diags.size());
		///push some diags into diags
		///not duplicating elements

		/// LLoop through new diags created by elemental_i elemental_j
		for(const auto &d: short_diags)
		{
//			auto equivalent_diags = d.all_cyclic_related_trace_products();
			int same_diagram=-1;
			bool found=false;
//			while(!found)
//			{

		///make all_short_diags an ordered list, and insert elements correctly,
		///using std::find or something.
			auto search_result = all_short_diags.find(d);

			if(search_result != all_short_diags.end())
			{
				search_result->coef = search_result->coef + d.coef;
				if(search_result->coef==0)
					all_short_diags.erase(search_result);
			}
			else
				all_short_diags.insert(d);

		}
//		wick_logger->debug("done adding new_diags, {:d} short_diags", all_short_diags.size());
	}///done looping through elementals

	//cout << "done with elementals" << endl;

	///double check for zero diags
	for(auto it = all_short_diags.begin(); it != all_short_diags.end(); it++)
		if( (*it).coef == 0)
			all_short_diags.erase(it--);

	for(const auto &sd: all_short_diags)
	{
		vector<Trace<QuarkLine>> trs;
		for(const auto &st: sd.traces)
		{
			vector<QuarkLine> qls;
			for(const auto &sql: st.qls)
			{
				qls.push_back(int_to_QL[sql]);
			}
			Trace<QuarkLine> t;
			t.qls=qls;
			trs.push_back(t);
		}
		diags.push_back(Diagram<QuarkLine>(sd.coef, trs));
	}
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

template <> void Correlator<QuarkLine>::load_numerical_results(Saved_Diagrams &computed)
{
	for(auto& d : diags)
  {
		for(auto& t : d.traces)
    {
      if( computed.count(t.name()) > 0 )
			{
        t.numerical_value = computed[t.name()];
		//		cout << t.name() << " has size " << computed[t.name()].size() << "," << flush;
		//		cout << computed[t.name()][0].size() << endl << flush;

		//		cout << "tr has size " << t.numerical_value.size() << "," << flush;
		//		cout << t.numerical_value[0].size() << endl << flush;
			}
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
			//for(const auto &ti : ts)
			//	for(const auto &dt : dts)
			//		if( t.numerical_value.count(to_string(dt)+" "+to_string(ti)) == 0 )
			//			throw 't';
		}///end of trace loop
	//	cout << "after trace loop " << d.traces[0].numerical_value.size() << endl << flush;
  }///end of diagram loop
	//cout << "after diag loop" << diags[0].traces[0].numerical_value.size() << endl << flush;
}



template <> void Correlator<QuarkLine>::compute_time_average_correlators()
{
//	cout << "made it here!" << endl;
//	cout << "dts.size() = " << dts.size() << "  ts.size()=" << ts.size() << endl;
  corr_t.resize(dts.size());
	for(size_t i=0; i<dts.size(); ++i)
  {
		int dt = dts[i];
    complex<double> time_avg(0.,0.);
    for(size_t i=0; i<ts.size(); ++i)
    {
			int t = ts[i];
      for(const auto& d : diags)
      {
        complex<double> trace_product(1.,0.);
        for(const auto& tr : d.traces)
        {
          trace_product *= tr.numerical_value[dt][t];
        }///end traces//
        time_avg += complex<double>(d.coef,0)*trace_product;
      }///end diags
    }///end t
    corr_t[i] = time_avg/(double(ts.size()));
  }///end dt

}
