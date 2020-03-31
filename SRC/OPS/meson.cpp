#include "OPS/meson.h"

#include "UTILS/string_utilities.h"

#include <sstream>

using namespace std;

Meson adjoint(Meson m)
{
	auto tmp = m.ql;
	m.ql=m.qr;
	m.qr=tmp;

	///flip momenta
	auto p = split(m.mom,' ');
	int px(stoi(p[0])), py(stoi(p[1])), pz(stoi(p[2]));

	stringstream ss;
	ss << -px << " " << -py << " " << -pz;
	m.mom = ss.str();

	return m;
}
