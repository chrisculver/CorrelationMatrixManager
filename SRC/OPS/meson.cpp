#include "OPS/meson.h"

Meson adjoint(Meson m)
{
	auto tmp = m.ql;
	m.ql=m.qr;
	m.qr=tmp;

	///flip momenta
	//
	return m;
}
