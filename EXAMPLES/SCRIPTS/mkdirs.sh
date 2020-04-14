file=rho_ops.txt

i=0
j=0

cat ${file} | while read line
do

	cat ${file} | while read line
	do
		mkdir OP${i}_OP${j}
		echo OP${i}_OP${j} >> dirnames.txt
		j=`expr $j + 1`
	done
	
	i=`expr $i + 1`
done
