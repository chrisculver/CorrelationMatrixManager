#ifndef DIAGS_DIAGRAM_H
#define DIAGS_DIAGRAM_H

#include "DIAGS/trace.h"

#include <string>
#include <vector>

template <class QL_Type>
 class Diagram
{
	public:
		std::vector<Trace<QL_Type>> traces;
		int coef;

	Diagram(){coef=0;};
	Diagram(int c, std::vector<Trace<QL_Type>> t):coef(c), traces(t){};

	std::string name() const;

	std::vector<std::vector<Trace<QL_Type>>> all_cyclic_related_diagrams() const
  {
  	///This finds ALL diagrams that are related by cyclic permutations.
  	/// [ A B ] [ C D ]
  	/// First we make all permutations of the traces, i.e.
  	/// { [ A B ] [ C D ] , [ C D ] [ A B ]}
  	/// For each of these lists we also need to permute each element w/in the trace
  	/// { [ A B ] [ C D ] , [ B A ] [C D ], ...   }

  	///the vector of traces the algorithm starts at
  	auto start = traces;
  	std::vector<std::vector<std::vector<Trace<QL_Type>>>> tmp(start.size()+1);

  	///all orders of the traces.
  	heaps_algorithm(tmp[0], start, start.size());
  	///for each order of traces, cyclic rotate each trace's elements getting
  	///all permutations.
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
};


#endif
