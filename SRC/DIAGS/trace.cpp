#include "DIAGS/trace.h"

bool operator==(const Trace &l, const Trace &r)
{
	return l.qls==r.qls;
}
