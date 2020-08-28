#ifndef DIAGS_CORRELATOR_H
#define DIAGS_CORRELATOR_H

#include "OPS/operator.h"
#include "DIAGS/diagram.h"

#include <vector>
#include <complex>
#include <map>
#include <string>

class Correlator
{
	public:
		const Operator c, a;

		std::vector<Diagram> diags;

		std::vector<std::complex<double>> corr_t;

		Correlator(Operator n_a, Operator c_a):c(c_a), a(n_a){};

		void wick_contract();
		void load_wick_contractions(const std::string filename, const int i, const int j);

		using Saved_Traces = std::map<std::string, std::vector<std::vector<std::complex<double>>> >;
		void load_numerical_results(Saved_Traces computed);

		void compute_time_average_correlators(int NT);
};

#endif
