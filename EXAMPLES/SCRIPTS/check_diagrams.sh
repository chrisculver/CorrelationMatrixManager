#!/bin/bash

latsize=28242464
dirname=/hpcgpfs01/work/lqcd/a1res/cculver/2864_diags_P100/

expected_lines=$((8118*24*64 + 8118))

###  2864_P000  expected_lines=$((2599*34))

echo ${expected_lines}
for i in {200..998..2}
do
        cfg=$(printf "%03d" ${i})
        file=${dirname}diags_${latsize}_${cfg}.dat
        if [ ! -f $file ]
        then
                echo "Config $cfg does not exist"
        else
                lines=$(wc -l $file | awk '{print $1}')

                if [[ ${lines} -ne ${expected_lines} ]]
                then
                        echo "Config $cfg has ${lines} lines instead of ${expected_lines}"
                fi
        fi
done
