#ifndef DIAGS_DIAGRAM_H
#define DIAGS_DIAGRAM_H

#include "DIAGS/trace.h"

#include <vector>

class Diagram
{
	public:
		std::vector<Trace> traces;
		int coef;
};

#endif
