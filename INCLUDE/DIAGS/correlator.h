#ifndef DIAGS_CORRELATOR_H
#define DIAGS_CORRELATOR_H

#include "OPS/operator.h"
#include "DIAGS/diagram.h"

#include <vector>
#include <complex>
#include <map>
#include <string>

template <class QL_Type>
class Correlator
{
	public:
		const Operator c, a;

		std::vector<Diagram<QL_Type>> diags;

		std::vector<std::complex<double>> corr_t;

		std::vector<int> ts, dts;

		Correlator(){};
		Correlator(Operator n_a, Operator c_a,
			std::vector<int> n_ts, std::vector<int> n_dts):c(c_a), a(n_a), ts(n_ts), dts(n_dts){};

		void wick_contract();
		void load_wick_contractions(const std::string filename, const int i, const int j);

		using Saved_Traces = std::map<std::string, std::map<std::string,std::complex<double>>>;
		void load_numerical_results(Saved_Traces computed);

		void compute_time_average_correlators();

};

#endif
