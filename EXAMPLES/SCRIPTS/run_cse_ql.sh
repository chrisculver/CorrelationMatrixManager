#!/bin/bash

#SBATCH -p defq -N 1
#SBATCH -e trash/slurm-out%j.log
#SBATCH -o trash/slurm-out%j.log
#SBATCH -t 5-00:00:00
#SBATCH -J rho_test

module load gcc/4.7.4

nx=24
nt=48
kappa=0.1282
mpi=300

### MODIFY THESE VARIABLES WITH PROJECTS AND DEVICES

dir=/lustre/groups/qcd/chris/A1_TEST/

cfg1=$(printf "%03d" ${1})
cfg2=$(printf "%03d" $(($1+1)))

cfgdir=/groups/qcd/configs/${nx}${nt}_${mpi}mev
latname=knhyp${nx}24${nt}_beta7.1_kappa${kappa}

echo ${cfg1}
echo ${cfgdir}/${latname}_${cfg1}

### NOW SUBMIT THE JOBS
OMP_NUM_THREADS=20 GOMP_CPU_AFFINITY="0-39:2" ./compute_diagrams_anyql <<EOF > ${dir}/logs/out_${cfg1}.log &
nx ${nx}
ny 24
nz 24
nt ${nt}
cfg ${cfg1}
nvec 100
ndiags 2807 
latname ${cfgdir}/${latname}_${cfg1}
unique_mom:length 7
unique_mom:0 0 0 0
unique_mom:1 -1 0 0
unique_mom:2 1 0 0
unique_mom:3 0 0 -1
unique_mom:4 0 0 1
unique_mom:5 0 -1 0
unique_mom:6 0 1 0
unique_gammas:length 4
unique_gammas:0 1 5
unique_gammas:1 1
unique_gammas:2 5
unique_gammas:3 6
unique_displacement:length 1
unique_displacement:0 \delta_{ii}
unique_displacement:length 1
unique_displacement:0 \delta_{ii}
EOF

OMP_NUM_THREADS=20 GOMP_CPU_AFFINITY="1-39:2" ./compute_diagrams_anyql <<EOF > ${dir}/logs/out_${cfg2}.log &
nx ${nx}
ny 24
nz 24
nt ${nt}
cfg ${cfg2}
nvec 100
ndiags 2807
latname ${cfgdir}/${latname}_${cfg2}
unique_mom:length 7
unique_mom:0 0 0 0
unique_mom:1 -1 0 0
unique_mom:2 1 0 0
unique_mom:3 0 0 -1
unique_mom:4 0 0 1
unique_mom:5 0 -1 0
unique_mom:6 0 1 0
unique_gammas:length 4
unique_gammas:0 1 5
unique_gammas:1 1
unique_gammas:2 5
unique_gammas:3 6
unique_displacement:length 1
unique_displacement:0 \delta_{ii}
unique_displacement:0 \delta_{ii}
EOF



wait


