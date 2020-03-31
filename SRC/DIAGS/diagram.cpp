#include "DIAGS/diagram.h"

using namespace std;

std::string Diagram::name() const
{
	std::string name(to_string(coef));
	for(const auto &t: traces)
	{
		name+="[ ";
		for(size_t i=0; i<t.qls.size(); ++i)
		{
			auto q=t.qls[i];
			name+=q.gamma + " " + q.displacement + " " + q.mom + " " + q.ti + " " + q.tf;
			if(i!=t.qls.size()-1)
				name+=" | ";
			else
				name+=" ";
		}
		name+="]";
	}

	return name;
}
