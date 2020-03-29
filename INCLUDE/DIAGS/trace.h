#ifndef DIAGS_TRACE_H
#define DIAGS_TRACE_H

#include "DIAGS/quark_line.h"

#include <vector>
#include <complex>

class Trace
{
	std::vector<QuarkLine> qls;

	std::vector<std::vector<std::complex<double>>> numerical_value;
};

#endif
