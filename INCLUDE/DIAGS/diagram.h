#ifndef DIAGS_DIAGRAM_H
#define DIAGS_DIAGRAM_H

#include "DIAGS/trace.h"

#include <string>
#include <vector>

class Diagram
{
	public:
		std::vector<Trace> traces;
		int coef;
	
	Diagram(){coef=0;};
	Diagram(int c, std::vector<Trace> t):coef(c), traces(t){};

	std::string name() const;

	std::vector<std::vector<Trace>> all_cyclic_related_diagrams() const;
};



#endif
