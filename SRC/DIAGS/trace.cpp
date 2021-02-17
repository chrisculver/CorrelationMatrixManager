#include "DIAGS/trace.h"

using namespace std;


template<> string Trace<QuarkLine>::name() const
{
	string name;
	name+="[ ";
	for(size_t i=0; i<qls.size(); ++i)
	{
		auto q=qls[i];
		name+=q.gamma + " " + q.displacement + " " + q.mom + " " + q.ti + " " + q.tf;
		if(i!=qls.size()-1)
			name+=" | ";
		else
			name+=" ";
	}
	name+="]";

	return name;
}


template<> vector<string> Trace<QuarkLine>::compute_name(vector<string> u_mom, vector<string> u_disp, vector<string> u_gamma)
{
  ///consider setting print options, then
  ///output for each quark line the string of how the quark line is
  ///stored, i.e. qf[ql_idx], qti[ql_idx] etc....
  vector<string> name(qls.size(),"");
  for(size_t i=0; i<qls.size(); ++i)
  {
    char t1(qls[i].ti), t2(qls[i].tf);
    if(t1=='i' && t2=='i')
      name[i]+="qti[";
    else if(t1=='i' && t2=='f')
      name[i]+="qf[";
    else if(t1=='f' && t2=='i')
      name[i]+="qb[";
    else if(t1=='f' && t2=='f')
      name[i]+="qtf[";

    for(size_t s=0; s<u_gamma.size(); ++s)
      if(qls[i].gamma==u_gamma[s])
        name[i]+=to_string(s) + "*" + to_string(u_disp.size()) + "*"
           + to_string(u_mom.size()) + " + ";
    for(size_t s=0; s<u_disp.size(); ++s)
      if(qls[i].displacement==u_disp[s])
        name[i]+=to_string(s) + "*" + to_string(u_mom.size()) + " + ";
    for(size_t s=0; s<u_mom.size(); ++s)
      if(qls[i].mom==u_mom[s])
        name[i]+=to_string(s) + "]";

  }

  return name;
}
