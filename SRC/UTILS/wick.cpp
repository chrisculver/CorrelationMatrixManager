#include "UTILS/wick.h"

#include "OPS/meson.h"

#include "spdlog/spdlog.h"

#include <iostream>
#include <map>

using namespace std;

vector<Diagram> wick_contract_elems(const ElementalOp &a, const ElementalOp &c)
{
	auto wick_logger = spdlog::get("wick");

	char label = 'a';
	map<char, Meson> meson_map;
	map<char, char> time_map;
	vector<ShortQuark> barred, unbarred;

	for(const auto &m: a.mesons)
	{
		barred.push_back( ShortQuark(true, m.ql, label) );
		unbarred.push_back( ShortQuark(false, m.qr, label) );
		meson_map.insert(pair<char, Meson>(label, m));
		time_map.insert(pair<char, char>(label, 'f'));
		label++;
	}
	for(const auto &m: c.mesons)
	{
		barred.push_back( ShortQuark(true, m.ql, label) );
		unbarred.push_back( ShortQuark(false, m.qr, label) );
		meson_map.insert(pair<char, Meson>(label, m));
		time_map.insert(pair<char, char>(label, 'i'));
		label++;
	}

//	wick_logger->debug("Beginning heaps algorithm");
	vector<vector<ShortQuark>> all_barred_permutations;
	vector<bool> all_signs;
	bool starting_sign = false;
	//cout << "Starting heaps algorithm, unbarred quark list is \n";
	//for(size_t i=0; i<unbarred.size(); ++i)
	//	cout << unbarred[i].barred << "-" << unbarred[i].flavor << "_" << unbarred[i].label << " ";
	//cout << endl;
	heaps_algorithm_anticommuting( all_barred_permutations, barred, barred.size(),
																 all_signs, starting_sign );
//	wick_logger->debug("Finished heaps algorithm");
	/*
	cout << "Printing all_permutations of the barred quarks\n";
//	if(spdlog::logger::level() == spdlog::level::debug)
//	{
		std::string all_permutes_string;
		for(size_t i=0; i<all_barred_permutations.size(); ++i)
		{
			const auto l = all_barred_permutations[i];
			all_permutes_string += to_string(all_signs[i]) + "  ";
			for(size_t j=0; j<l.size(); ++j)
			{
				const auto qbar = l[j];
				const auto q = unbarred[j];
				all_permutes_string += to_string(qbar.barred)+"-"+qbar.flavor+"_"+qbar.label+" ";
				all_permutes_string += to_string(q.barred)+"-"+q.flavor+"_"+q.label+" ";

			}
			all_permutes_string += "\n";
		}
		cout << all_permutes_string << endl;
		//wick_logger->debug("Printing all_barred_permutations\n{}", all_permutes_string);
//	}
	cout << "printed all permutations\n";
	*/
	vector<Diagram> res;
	for(size_t i=0; i<all_barred_permutations.size(); ++i)
	{
		auto lst = all_barred_permutations[i];
    bool contractable = true;
		for(size_t q=0; q<unbarred.size(); ++q)
			if(unbarred[q].flavor != lst[q].flavor)
				contractable = false;


		if(contractable)
		{
			Diagram d;
			Trace t;
			vector<bool> q_contracted(unbarred.size(), false);
			bool all_contracted = false;
			bool new_trace_loop = false;
			int q = 0;///current quark we are on, start at beginning
			///Start at quark q=0.  Make the quark line for unbarred to barred.
			///Find the quark it goes to next by label.
			///Exit when all quarks are used
			char starting_meson = unbarred[q].label;
			while(!all_contracted)
			{
//				cout << "q=" << q << "  |  m.label=" << lst[q].label << "  |  starting_meson=" << starting_meson << endl;
				Meson m = meson_map[unbarred[q].label];
				t.qls.push_back( QuarkLine(time_map[unbarred[q].label], m, time_map[lst[q].label]) );
				q_contracted[q]=true;

				if(lst[q].label == starting_meson)
				{
					d.traces.push_back(t);
					t.qls.clear();
					new_trace_loop = true;
				}

				all_contracted=true;
				for(size_t i=0; i<q_contracted.size(); ++i)
					all_contracted = all_contracted && q_contracted[i];

				///Find the meson we are attached to.
				if(new_trace_loop)
				{
					for(size_t i=0; i<unbarred.size(); ++i)
						if(q_contracted[i]==false)
						{
							starting_meson = unbarred[i].label;
							q=i;
							new_trace_loop=false;
							break;
						}
				}
				else
				{
					for(size_t i=0; i<unbarred.size(); ++i)
						if(lst[q].label == unbarred[i].label)
						{
							q=i;
							break;
						}
				}
			}

//			cout << endl;
		  if(all_signs[i])
        d.coef=-1*a.coef*c.coef;
      else
        d.coef=a.coef*c.coef;

			///We used a list \bar{q} q \cdots \bar{q} q but props are
			///formed from q \bar{q}, -1^N_mesons factor needs to be applied
			int tot_mesons = a.mesons.size() + c.mesons.size();
			//cout << "tot_mesons = " << tot_mesons << "   " << (tot_mesons%2) << endl;
			if( (tot_mesons%2) != 0)
				d.coef*=-1;

      res.push_back(d);
		}
	}

//	cout << "The following diagrams are being added to the correlator\n";
//	for(const auto &d: res)
//	{
//		cout << d.name() << "\n";
//	}

	return res;
}
