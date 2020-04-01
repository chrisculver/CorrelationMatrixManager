#ifndef DIAGS_TRACE_H
#define DIAGS_TRACE_H

#include "DIAGS/quark_line.h"

#include <vector>
#include <complex>

class Trace
{
	public:
		std::vector<QuarkLine> qls;

		std::vector<std::vector<std::complex<double>>> numerical_value;

};

bool operator==(const Trace &l, const Trace &r);

#endif
