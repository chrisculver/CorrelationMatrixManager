#!/bin/bash

NC=299

while read dirname
do

	lst=()
	folder=${dirname}

	for file in ${folder}/*.dat
	do
		tmp="$(basename "$file")"
		lst+=(${tmp::-8})
	done 

	lst=($(printf "%s\n" "${lst[@]}" | sort | uniq -c | sort -rnk1 | awk '{ print $2 }'))
	echo "${uniq[@]}"

#	for i in {0..${NC}..1}
	for ((i=0; i<=${NC}; ++i))
	do
		cfg=$(printf "%03d" ${i})
		suffix="_${cfg}.dat"
		for f in "${lst[@]}"
		do
			cat "${folder}/${f}_${cfg}.dat" >> "${f}.dat"
		done

	done

done < dirnames.txt
