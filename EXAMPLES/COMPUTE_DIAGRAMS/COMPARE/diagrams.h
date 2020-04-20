#ifndef REDUCTION_H
#define REDUCTION_H

#include <string>
#include <vector>

class Quark_Line
{
  public:
    Quark_Line(){};
    Quark_Line(std::string gamma_n, std::string mom_n, std::string time_n);
    Quark_Line(const Quark_Line &q1){gamma=q1.gamma;mom=q1.mom;t=q1.t;};
    std::string get_gamma();
    std::string get_mom();
    std::string get_time();
    
    void set_mom(std::string mom_n){mom=mom_n;};
    void set_time(std::string time_n){t=time_n;};

    friend std::ostream &operator<<(std::ostream &stream, const Quark_Line &q);
    bool operator==(const Quark_Line &q1) const; 

  private:
    std::string gamma;
    std::string mom;
    std::string t;
};

class Trace
{
  public:
    Trace(){};
    Trace(std::vector<Quark_Line> q_n):q(q_n){};
    Trace(const Trace &t){q=t.q;};

    friend std::ostream &operator<<(std::ostream &stream, const Trace &tr); 
    bool operator==(const Trace &t1) const;
    
    void update_mom(std::vector<std::string> new_mom, std::vector<std::string> old_mom);
    void swap_time();

    void swap_mom_time_two();
    void swap_mom_time_four_right();
    void fix_adjoint();

    std::vector<Quark_Line> get_q(){return q;};
  private:
    std::vector<Quark_Line> q;
};

bool is_cyclic_permutation(std::vector<Quark_Line> q1, std::vector<Quark_Line> q2);
void cyclic_rotate(std::vector<Trace> &tr);
void cyclic_rotate_timerev(std::vector<Trace> &tr);
bool is_boson_swap(std::vector<Quark_Line> q1, std::vector<Quark_Line> q2, int rot);
void boson_rotate(std::vector<Trace> &tr);
void all_reductions(std::vector<Trace> &tr);


#endif
