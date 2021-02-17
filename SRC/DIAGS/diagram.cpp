#include "DIAGS/diagram.h"
#include "UTILS/wick.h"

#include <algorithm>
#include <iostream>

using namespace std;


template <> std::string Diagram<QuarkLine>::name() const
{
	std::string name(to_string(coef));
	for(const auto &t: traces)
	{
		name+="[ ";
		for(size_t i=0; i<t.qls.size(); ++i)
		{
			auto q=t.qls[i];
			name+=q.gamma + " " + q.displacement + " " + q.mom + " " + q.ti + " " + q.tf;
			if(i!=t.qls.size()-1)
				name+=" | ";
			else
				name+=" ";
		}
		name+="]";
	}

	return name;
}


template <> vector<vector<Trace<QuarkLine>>> Diagram<QuarkLine>::all_cyclic_related_diagrams() const
{
	///This finds ALL diagrams that are related by cyclic permutations.
	/// [ A B ] [ C D ]
	/// First we make all permutations of the traces, i.e.
	/// { [ A B ] [ C D ] , [ C D ] [ B A ]}
	/// For each of these lists find

	///the vector of traces the algorithm starts at
	auto start = traces;
	vector<vector<vector<Trace<QuarkLine>>>> tmp(start.size()+1);

	heaps_algorithm(tmp[0], start, start.size());


	for(size_t i=0; i<start.size(); ++i)
	{

		for(size_t lst=0; lst<tmp[i].size(); ++lst)
		for(size_t j=0; j<tmp[i][lst][i].qls.size(); ++j)
		{
			rotate(tmp[i][lst][i].qls.begin(), tmp[i][lst][i].qls.begin()+1, tmp[i][lst][i].qls.end());
			tmp[i+1].push_back(tmp[i][lst]);
		}

	}

	return tmp[start.size()];
}
