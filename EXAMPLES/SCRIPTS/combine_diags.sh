#!/bin/bash
#SBATCH -p short,defq -N 1
#SBATCH -e trash/slurm-out%j.log
#SBATCH -o trash/slurm-out%j.log
#SBATCH -t 6:00:00
#SBATCH -J 2448_DIAGS

module load gcc/6.5.0

let start=0+${1}
let end=39+${1}

nx=24
nt=48

opfile=rho_ops.txt

for i in $(eval echo "{$start..$end..1}")
do

OMP_NUM_THREADS=${i} ./compute_correlator <<EOF > out_test.dat &
nx ${nx}
ny 24
nz 24
nt ${nt}
cfg ${i}
operator_filename ${opfile}
wick_directory /CCAS/home/chrisculver/JOBS/RHO/GenCPP/
numerical_directory /CCAS/home/chrisculver/JOBS/RHO/GenCPP/
gpu_memory 4096
EOF

done

wait


i=0
j=0

cat ${opfile} | while read line
do

	cat ${opfile} | while read line
	do
		mv corr_o1.${i}_o2.${j}_*.dat OP${i}_OP${j}/
		j=`expr $j + 1`
	done

	i=`expr $i + 1`
done
