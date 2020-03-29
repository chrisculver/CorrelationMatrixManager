#ifndef OPS_MESON_H
#define OPS_MESON_H

#include <string>

struct Meson
{
	///ql is the left quark an anti-quark, qr is a quark
	char ql, qr;	
	///The structure that determines Gamma(p).
	std::string gamma, displacement, mom;

	Meson(char l, std::string g, std::string d, std::string m, char r):
		ql(l), gamma(g), displacement(d), mom(m), qr(r){};

	template<typename OStream>
	friend OStream &operator<<(OStream &os, const Meson &m)
	{
		return os << m.ql << "," << m.gamma << "," << m.displacement << "," << m.mom << "," << m.qr;
	}

};
	
Meson adjoint(Meson m);

#endif
