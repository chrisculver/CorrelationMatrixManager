#include "DIAGS/diagram.h"
#include "UTILS/wick.h"

#include <algorithm>
#include <iostream>

using namespace std;

std::string Diagram::name() const
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


vector<vector<Trace>> Diagram::all_cyclic_related_diagrams() const
{
	vector<vector<Trace>> outer_permuted, res;
	auto rot = traces;
//	res.push_back(traces);
//	cout << "finding all related diags of " << endl;
//	cout << name() << endl << endl;

	heaps_algorithm(outer_permuted, rot, rot.size());

	for(size_t i=0; i<outer_permuted.size(); ++i)
	{
		for(size_t j=0; j<outer_permuted[i].size(); ++j)
		{
			for(size_t k=0; k<outer_permuted[i][j].qls.size(); ++k)
			{
				rotate(outer_permuted[i][j].qls.begin(), outer_permuted[i][j].qls.begin()+1, outer_permuted[i][j].qls.end());
				res.push_back(outer_permuted[i]);
			}
		}
	}


//
//	for(size_t i=0; i<rot.size(); ++i)
//	{
//		rotate(rot.begin(), rot.begin()+1, rot.end());
//		for(size_t j=0; j<rot.size(); ++j)
//		{
//			for(size_t k=0; k<rot[j].qls.size(); ++k)
//			{
//				res.push_back(rot);
//				rotate(rot[j].qls.begin(), rot[j].qls.begin()+1, rot[j].qls.end());
//			}
//		}
//	}
	
	
	for(size_t i=0; i<res.size(); ++i)
	{
//		for(size_t j=0; j<res[i].size(); ++j)
//			cout << res[i][j].name() << " ";
//		cout << endl;
	}

	return res;
}
