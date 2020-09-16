#ifndef DIAGS_QUARK_LINE_H
#define DIAGS_QUARK_LINE_H

#include "OPS/meson.h"

#include <string>


/*!
 *   Quark lines are defined as Gamma(p)*M^{-1}.
 *   The meson for Gamma(p) corresponds to t_i, the starting time of the propagator.
 */
class QuarkLine
{
	public:
		char flavor; ///flavor of the propagator , 'l'/'s'/'c'...
		std::string gamma;
		std::string displacement;
		std::string mom;
		char ti, tf;

		QuarkLine(){};
		QuarkLine(char i, char fl, std::string g, std::string d, std::string m, char f):
			ti(i),flavor(fl),gamma(g),displacement(d),mom(m),tf(f){};
///   deprecated until I figure out whether the QL prop comes from the
///   bar or unbarred quark.
///		QuarkLine(char i, Meson m, char f):
///			ti(i),gamma(m.gamma), displacement(m.displacement), mom(m.mom), tf(f){};

};

bool operator==(const QuarkLine &l, const QuarkLine &r);

#endif
