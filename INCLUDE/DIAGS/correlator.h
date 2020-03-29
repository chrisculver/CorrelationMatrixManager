#ifndef DIAGS_CORRELATOR_H
#define DIAGS_CORRELATOR_H

#include "OPS/operator.h"
#include "DIAGS/diagram.h"

#include <vector>
#include <complex>

class Correlator
{
	public:
		const Operator c, a;

		std::vector<Diagram> diags;

		std::vector<std::complex<double>> corr_t;

		Correlator(Operator n_a, Operator c_a):c(c_a), a(n_a){};

		void wick_contract();
};

#endif
