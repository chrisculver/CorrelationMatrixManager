#include "DIAGS/quark_line.h"

bool operator==(const QuarkLine &l, const QuarkLine &r)
{
	return (l.gamma==r.gamma) && (l.displacement==r.displacement) &&
		     (l.mom==r.mom) && (l.ti==r.ti) && (l.tf==r.tf);
}

bool operator<(const QuarkLine &l, const QuarkLine &r)
{
	return (l.fullname < r.fullname);
}
