#include "DIAGS/trace.h"

using namespace std;

bool operator==(const Trace &l, const Trace &r)
{
	return l.qls==r.qls;
}

string Trace::name() const
{
	string name;
	name+="[ ";
	for(size_t i=0; i<qls.size(); ++i)
	{
		auto q=qls[i];
		stringstream tmp;
		tmp << q.flavor << " " << q.gamma << " " << q.displacement << " " << q.mom << " " << q.ti << " " << q.tf;
		name += tmp.str();
		if(i!=qls.size()-1)
			name+=" | ";
		else
			name+=" ";
	}
	name+="]";

	return name;
}


vector<string> Trace::compute_name(vector<string> u_mom, vector<string> u_disp, vector<string> u_gamma)
{
  ///consider setting print options, then
  ///output for each quark line the string of how the quark line is
  ///stored, i.e. qf[ql_idx], qti[ql_idx] etc....
  vector<string> name(qls.size(),"");
  for(size_t i=0; i<qls.size(); ++i)
  {
    char t1(qls[i].ti), t2(qls[i].tf), fl(qls[i].flavor);
		stringstream tmp;
		if(t1=='i' && t2=='i')
      tmp << "qti";
    else if(t1=='i' && t2=='f')
      tmp << "qf";
    else if(t1=='f' && t2=='i')
      tmp << "qb";
    else if(t1=='f' && t2=='f')
      tmp << "qtf";

		tmp << "_" << fl << "[";
		name[i]+= tmp.str();

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
