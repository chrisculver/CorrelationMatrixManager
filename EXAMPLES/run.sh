#!/bin/bash

in_file=a1_input.txt

./compute_correlation_matrix ${in_file}
res=$?

echo $res

if [ "$res" -eq 1 ]
then
	echo "Computing diagrams on 4x4"
	cp -r ../EXAMPLES/COMPUTE_DIAGRAMS .
	mv define_diagrams.cpp COMPUTE_DIAGRAMS
	mv diagram_names.txt COMPUTE_DIAGRAMS
	mv run_c44.in COMPUTE_DIAGRAMS
	cd COMPUTE_DIAGRAMS
	make compute_diagrams_anyql
	./compute_diagrams_anyql < run_c44.in
	mv diags_4444_100.dat ..
	cd ..
	./compute_correlation_matrix ${in_file}

elif [ "$res" -eq 0 ]
then
	echo "Success\n"	
fi

