#ifndef UTILS_WICK_H
#define UTILS_WICK_H

#include "DIAGS/diagram.h"
#include "OPS/elemental_op.h"

#include <iostream>
#include <vector>

struct ShortQuark
{
	///Quark or anti-quark
	bool barred;
	///Quark flavor
	char flavor;
	///label supressing all other indices
	char label;
	///So that struct is 4 bytes.  
	char Padding;

	ShortQuark(bool b, char f, char l):barred(b), flavor(f), label(l){};

	bool friend operator==(const ShortQuark &q1, const ShortQuark &q2)
	{
		return (q1.barred==q2.barred) && (q1.flavor==q2.flavor) && (q1.label==q2.label);
	}
};

std::vector<Diagram> wick_contract_elems(const ElementalOp &a, const ElementalOp &c);


template <class T>
void heaps_algorithm(
		std::vector< std::vector<T> > &dest, std::vector<T> &src, int size
		)
{
	if(size==1)
		dest.push_back(src);

	for(int i=0; i<size; ++i)
	{
		heaps_algorithm(dest, src, size-1);
		if(size%2==1)
			iter_swap(src.begin(), src.begin() + size - 1);
		else
			iter_swap(src.begin() + i, src.begin() + size - 1);
	}
}


template <class T>
void heaps_algorithm_anticommuting(
		std::vector< std::vector<T> > &dest, std::vector<T> &src, int size, 
		std::vector<bool> &coef, bool &c)
{
	
	if(size==1)
	{
		dest.push_back(src);
		coef.push_back(c);
	}

	for(int i=0; i<size; ++i)
	{
		heaps_algorithm_anticommuting(dest, src, size-1, coef, c);
		std::vector<T> old_src = src;

/*		using std::cout; using std::endl;
		cout << "List before swap\n";
		cout << c << " ";
		for(size_t i=0; i<src.size(); ++i)
			cout << src[i].barred << "-" << src[i].flavor << "_" << src[i].label << " ";
		cout << endl;
*/
		if(size%2==1)
		{
			iter_swap(src.begin(), src.begin() + size - 1);
		}
		else
		{
			iter_swap(src.begin() + i, src.begin() + size - 1);
		}
		if( src != old_src)
			c=!c;
/*	
		cout << "List after swap\n";
		cout << c << " ";
		for(size_t i=0; i<src.size(); ++i)
			cout << src[i].barred << "-" << src[i].flavor << "_" << src[i].label << " ";
		cout << endl << endl << endl;
*/
	}
}



#endif
