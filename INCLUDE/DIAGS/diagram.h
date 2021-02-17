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

	std::vector<std::vector<Trace<QL_Type>>> all_cyclic_related_diagrams() const;
};



#endif
