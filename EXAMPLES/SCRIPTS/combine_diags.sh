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

opfile=a1_2448_test.ops

for i in $(eval echo "{$start..$end..1}")
do

echo "nx ${nx}" >> a1_input_${i}.in
echo "ny 24" >> a1_input_${i}.in
echo "nz 24" >> a1_input_${i}.in
echo "nt ${nt}" >> a1_input_${i}.in
echo "cfg ${i}" >> a1_input_${i}.in
echo "verbose_logging 0" >> a1_input_${i}.in
echo "operator_filename ${opfile}" >> a1_input_${i}.in

OMP_NUM_THREADS=${i} ./compute_correlation_matrix a1_input_${i}.in &

done


wait

for i in $(eval echo "{$start..$end..1}")
do
	rm a1_input_${i}.in
done

i=0
j=0

cat ${opfile} | while read line
do

	cat ${opfile} | while read line
	do
		mv corr_op.${i}_op.${j}_*.dat OP${i}_OP${j}/
		j=`expr $j + 1`
	done

	i=`expr $i + 1`

done
