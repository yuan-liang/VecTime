#! /bin/bash



total=31;
a=7;
b=24;

for exe in  1d-gs-3p
do
    rm  -f result/$exe.log
    for index in `seq $a $b`;
    do 
    nx=$((2 ** $index))
    t=$((2 ** ($total - $index)))
    ../$exe/exe-$exe $nx $t  |grep GS | tee -a result/$exe.log
    done
done

echo -e "size\tscalar\tour" | tee result/$exe.data.txt
cat result/$exe.log|tr "," "=" | awk -F "= " '{print $3 "\t" $NF}'|paste  - - | awk '{print $1"\t" $2"\t" $4"\t"}' |tee -a result/$exe.data.txt
