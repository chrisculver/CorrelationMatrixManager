#include "UTILS/wick.h"

#include "OPS/meson.h"

#include "spdlog/spdlog.h"

#include <iostream>
#include <map>

using namespace std;

vector<Diagram> wick_contract_elems(const ElementalOp &a, const ElementalOp &c)
{
	auto wick_logger = spdlog::get("wick");
	wick_logger->debug("Contracting elementals");
	
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

	wick_logger->debug("Beginning heaps algorithm");
	vector<vector<ShortQuark>> all_barred_permutations;
	vector<bool> all_signs;
	heaps_algorithm_anticommuting( all_barred_permutations, barred, barred.size(), 
																 all_signs, true );
	wick_logger->debug("Finished heaps algorithm");
	
//	cout << "Printing all_permutations of the barred quarks\n";
//	if(spdlog::logger::level() == spdlog::level::debug)
//	{
		std::string all_permutes_string;
		for(const auto &l: all_barred_permutations)
		{
			for(const auto &q: l)
				all_permutes_string += to_string(q.barred)+"-"+q.flavor+"_"+q.label+" ";
			all_permutes_string += "\n";
		}
//		cout << all_permutes_string << endl;
		//wick_logger->debug("Printing all_barred_permutations\n{}", all_permutes_string);
//	}
//	cout << "printed all permutations\n";
	
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
			cout << "quark_list = ";
			char starting_meson = unbarred[0].label;
			Diagram d;
			Trace t;
			for(size_t q=0; q<unbarred.size(); ++q)
			{
				cout << unbarred[q].barred << "-" << unbarred[q].flavor << "_" << unbarred[q].label << " ";
				cout << lst[q].barred << "-" << lst[q].flavor << "_" << lst[q].label << " ";

				Meson m = meson_map[unbarred[q].label];
				t.qls.push_back(QuarkLine( time_map[unbarred[q].label], m, time_map[lst[q].label] ));
				
				if(lst[q].label == starting_meson)
				{
					d.traces.push_back(t);
					t.qls.clear();
					starting_meson = unbarred[(q+1)%unbarred.size()].label;///does the mod need to be there?
				}
			}
			cout << endl;
		  if(all_signs[i])
        d.coef=1;
      else
        d.coef=-1;  
      res.push_back(d);
		}
	}		
	
	cout << "The following diagrams are being added to the correlator\n";
	for(const auto &d: res)
	{
		cout << d.name() << "\n";
	}


	return res;
}
