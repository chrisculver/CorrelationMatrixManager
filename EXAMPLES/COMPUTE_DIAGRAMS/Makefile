# Makefile
# Andrei Alexandru
# March 2011


#CC := /usr/local/Cellar/gcc/9.2.0_3/bin/gcc-9 --compile
#CPP := /usr/local/Cellar/gcc/9.2.0_3/bin/g++-9 --compile
#LD := /usr/local/Cellar/gcc/9.2.0_3/bin/g++-9

GQDIR = ${HOME}/Code/Libraries/gwu-qcd
include ${GQDIR}/Makefile.config
#include ${GQDIR}/.config

include ${GQDIR}/configs/arch/${ARCH}
#include ${GQDIR}/configs/arch/macosx

include ${GQDIR}/Makefile.local
LDFLAGS := -L${GQDIR}/lib -lqcd ${LDFLAGS}
CFLAGS += -I${GQDIR}/include -I ~/Code/Libraries/eigen -std=c++11
NVCC_FLAGS += -I${GQDIR}/include -I ~/Code/Libraries/eigen -w 


compute_diagrams : SRC/gammas.o SRC/io.o SRC/mom_matrix.o SRC/smeared_matrix_object.o SRC/laplacean_eigensolver.o SRC/lapmult.o SRC/inject.o SRC/colorvector.o SRC/gamma_mult.o define_diagrams.o
compute_diagrams_ql : SRC/gammas.o SRC/io.o SRC/mom_matrix.o SRC/smeared_matrix_object.o SRC/laplacean_eigensolver.o SRC/lapmult.o SRC/inject.o SRC/colorvector.o SRC/gamma_mult.o define_diagrams.o
compute_diagrams_anyql : SRC/gammas.o SRC/io.o SRC/mom_matrix.o SRC/smeared_matrix_object.o SRC/laplacean_eigensolver.o SRC/lapmult.o SRC/inject.o SRC/colorvector.o SRC/gamma_mult.o define_diagrams.o



