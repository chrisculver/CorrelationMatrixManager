#ifndef DIAGS_TRACE_H
#define DIAGS_TRACE_H

#include "DIAGS/quark_line.h"

#include <vector>
#include <complex>
#include <string>

class Trace
{
	public:
		std::vector<QuarkLine> qls;

		std::vector<std::vector<std::complex<double>>> numerical_value;

		std::string name() const;
		std::vector<std::string> compute_name(std::vector<std::string> u_mom, std::vector<std::string> u_disp, std::vector<std::string> u_gamma);	
};

bool operator==(const Trace &l, const Trace &r);

#endif
