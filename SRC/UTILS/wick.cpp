#include "UTILS/wick.h"

#include "OPS/meson.h"

#include <map>

using namespace std;

vector<Diagram> wick_contract_elems(const ElementalOp &a, const ElementalOp &c)
{
	vector<Diagram> res;
	
	char label = 'a';
	map<char, Meson> meson_map;
	vector<ShortQuark> barred, unbarred;

	for(const auto &m: a.mesons)
	{
		barred.push_back( ShortQuark(true, m.ql, label) );
		unbarred.push_back( ShortQuark(false, m.qr, label) );	
		label++;
	}	
	for(const auto &m: c.mesons)
	{
		barred.push_back( ShortQuark(true, m.ql, label) );
		unbarred.push_back( ShortQuark(false, m.qr, label) );
		label++;
	}

	vector<vector<ShortQuark>> all_barred_permutations;
	vector<bool> all_signs;
	heaps_algorithm_anticommuting( all_barred_permutations, barred, barred.size(), 
																 all_signs, true );



	return res;
}
