# README for SU(4) Meson Spectroscopy

This is a quick read-me on how to take a list of (multi)meson operators and run them through the SU(4) workflow stack.  Prerequisites for this document are a list of operators and stored eigenvectors/perambulators that you want to compute the correlation functions on.  Some operator files are provided in EXAMPLES/INPUT_FILES

This specific code reads in the operators and outputs a list of diagrams as well as pure cpp code to compute them using the Observables folder of https://gitlab.com/Kimmy.Cushman/laph_build_and_run using the mesons branch.  

Specific steps are now given.

1.  First install spdlog from https://github.com/gabime/spdlog
2.  Change the include_directories and find_library in CMakeLists.txt to where you build spdlog.
3.  Build this code with 
	mkdir build && cd build
	cmake ..
	make -j
4.  Now creat a .in and .ops file following the convention in the examples directory.
5.  Run the code with ./compute_correlation_matrix input.in
6.  Build laph_build_and_run by following the readme there, up until building the Observables directory
7.  Copy all the cpp files output from step 4 into laph_build_and_run/Observables/src
8.  Copy run_c44.in and diagram_names.txt into laph_build_and_run/Observables
9.  You should now be able to follow in the instructions in Observables/README.md
10. Successfully running the code there will create a cpu/gpu diagram output file.  Go make to where you ran ./compute_correlation_matrix and create a file d_files.txt with a line for the diagram output file.
11. Rerun the correlation_matrix_manager which should now create correlator output files. 
