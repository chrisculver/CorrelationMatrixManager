#include "diagrams.h"
#include "string_utilities.h"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>
#include <fstream>

using namespace std;

string Quark_Line::get_gamma(){return gamma;}
string Quark_Line::get_mom(){return mom;}
string Quark_Line::get_time(){return t;}

Quark_Line::Quark_Line(string gamma_n, string mom_n, string time_n)
{
  gamma=gamma_n;
  mom=mom_n;
  t=time_n;
}

bool Quark_Line::operator==(const Quark_Line &q1) const
{
  return ( (t==q1.t) && (mom==q1.mom) && (gamma==q1.gamma) );
}

ostream &operator<<(ostream &stream, const Quark_Line &q)
{
  stream << q.gamma << " " << q.mom << " " << q.t;
  return stream;
}

void Trace::update_mom(vector<string> new_mom, vector<string> old_mom)
{
  for(size_t i=0; i<q.size(); ++i){
    for(size_t j=0; j<new_mom.size(); ++j){
      if(q[i].get_mom() == old_mom[j])
        q[i].set_mom(new_mom[j]);
    }
  }
}

void Trace::swap_time()
{///only works for two times t and tf
  for(size_t i=0; i<q.size(); ++i){
    if(q[i].get_time()=="t")
      q[i].set_time("t_f");
    else if(q[i].get_time()=="t_f")
      q[i].set_time("t"); 
  }
}


void Trace::fix_adjoint()
{
  for(size_t i=0; i<q.size(); ++i)
  {
    if(q[i].get_time()=="t_f")
    {
      auto old_mom = split(q[i].get_mom(),' ');
      for(size_t j=0; j<old_mom.size(); ++j)
        if(stoi(old_mom[j])!=0)
          old_mom[j]=to_string(-1*stoi(old_mom[j]));
          
      q[i].set_mom(old_mom[0]+" "+old_mom[1]+" "+old_mom[2]);
    }
  }
}

void Trace::swap_mom_time_two()
{
  if(q.size()!=2)
  {
    cout << "ERROR BAD SWAP";
    return;
  }
  else
  {
    auto tmp = q[0].get_mom();
    q[0].set_mom(q[1].get_mom());
    q[1].set_mom(tmp);
  }
}

void Trace::swap_mom_time_four_right()
{
  if(q.size()!=4)
  {
    cout << "ERROR BAD SWAP";
    return;
  }
  else
  {
    auto qcopy = q;
    q[0].set_mom(qcopy[3].get_mom());
    q[1].set_mom(qcopy[0].get_mom());
    q[2].set_mom(qcopy[1].get_mom());
    q[3].set_mom(qcopy[2].get_mom());
  }
}

bool Trace::operator==(const Trace &t1) const
{
  return q==t1.q;
}

ostream &operator<<(ostream &stream, const Trace &tr)
{
  stream << "[ ";
  for(size_t i=0; i<tr.q.size(); ++i)
  {
    stream << tr.q[i];
    if(i==tr.q.size()-1)
      stream << " ]";
    else 
      stream << " | ";
  }
  return stream;
}

ostream &operator<<(ostream &stream, const vector<Trace> &t)
{
  for(size_t i=0; i<t.size(); i++)
    stream << t[i] << "\n";
  stream << "\n";
  return stream;
}

bool is_cyclic_permutation(vector<Quark_Line> q1, vector<Quark_Line> q2)
{
  bool permutation = false;
  int rotations = q1.size();
  for(int r=0; r<rotations; r++)
  {
    rotate(q2.begin(), q2.begin()+r, q2.end());
    if( q1 == q2 )
      permutation = true;
  }
  return permutation;
}

void cyclic_rotate(vector<Trace> &tr)
{
  int size=tr.size();
  for(int i=0; i<size; ++i){  
    bool del=false;
    for(int j=0; j<size; ++j){
      if(j!=i && (tr[i].get_q().size()==tr[j].get_q().size()) ) ///compare lists of similar size
      {
//        vector<Quark_Line> tmpi, tmpj; //need these
        auto tmpi=tr[i].get_q();
        auto tmpj=tr[j].get_q();

        if( is_cyclic_permutation(tmpi, tmpj) )
          del=true;
      }
    }
    if(del)
    {  
      tr.erase(tr.begin()+i);
      i--;
      size=tr.size();
    }
  }
}

void cyclic_rotate_timerev(vector<Trace> &tr) //only 2 mesons -> 2 mesons
{
  int size=tr.size();
  for(int i=0; i<size; ++i){  
    bool del=false;
    for(int j=0; j<size; ++j)
    {
      auto i_size = tr[i].get_q().size();
      auto j_size = tr[j].get_q().size();
      if(j!=i && (i_size==j_size) && (i_size<5) ) ///compare lists of similar size
      {
        auto tmpi=tr[i].get_q();
        Trace tmp(tr[j]);
        tmp.swap_time();
        auto tmpj=tmp.get_q();
        
        if( is_cyclic_permutation(tmpi,tmpj) )
          del=true;
      }
    }
    if(del)
    {  
      tr.erase(tr.begin()+i);
      i--;
      size=tr.size();
    }
  }
}


bool is_boson_swap(vector<Quark_Line> q1, vector<Quark_Line> q2, int rot)
///Definitely works for isospin 3 since pions at t and tf ALWAYS alternate
///WARNING MAY not work for other isospins.
{
  bool swap = false;
 
  auto test = q2;
  for(int i=0; i<test.size()/rot; ++i){
    rotate(test.begin(), test.begin() + rot, test.end());
    
    if( test == q1 )
      swap = true;
  }
  
  return swap;
}


void boson_rotate(vector<Trace> &tr)
{
  int size=tr.size();
  for(int i=0; i<size; ++i){  
    bool del=false;
    for(int j=0; j<size; ++j){
      if(j!=i && (tr[i].get_q().size()==tr[j].get_q().size()) ) ///compare lists of similar size
      {
        auto tmpi=tr[i].get_q();
        auto tmpj=tr[j].get_q();
        
        if( is_boson_swap(tmpi,tmpj, 2) )
          del=true;
        else if( is_boson_swap(tmpi, tmpj, 4) )  ///check numbers depends on nmesons?
          del=true;
      }
    }
    if(del)
    {  
      tr.erase(tr.begin()+i);
      i--;
      size=tr.size();
    }
  }

}


void all_reductions(vector<Trace> &tr)
{
//  cyclic_rotate(tr);
  /// Now check time reversal symmetries
  //cyclic_rotate_timerev(tr);

/// apparently redundant.
//  boson_rotate(tr);

}



