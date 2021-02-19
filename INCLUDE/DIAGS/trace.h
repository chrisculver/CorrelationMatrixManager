#ifndef DIAGS_TRACE_H
#define DIAGS_TRACE_H

#include "DIAGS/quark_line.h"

#include <vector>
#include <complex>
#include <map>
#include <string>

template <class QL_Type>
class Trace
{
	public:
		std::vector<QL_Type> qls;

		std::map<std::string, std::complex<double>> numerical_value;

		std::string name() const;
		std::vector<std::string> compute_name(std::vector<std::string> u_mom, std::vector<std::string> u_disp, std::vector<std::string> u_gamma);
};

template<class QL_Type> bool operator==(const Trace<QL_Type> &l, const Trace<QL_Type> &r)
{
	return l.qls==r.qls;
}

/*
template <class int>
class Trace
{
	public:
		std::vector<int> qls;
}
*/

#endif
