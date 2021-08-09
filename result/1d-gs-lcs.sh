#! /bin/bash




a=7;
b=18;

for exe in  1d-gs-lcs
do
    rm -f result/$exe.log
    for index in `seq $a $b`;
    do 
    nx=$((2 ** $index))
    ../$exe/exe-$exe $nx $nx | tee -a result/$exe.log 
    done
done

echo -e "size\tscalar\tour" | tee result/$exe.data.txt
cat result/$exe.log|tr "," "=" | awk -F "= " '{print $3 "\t" $NF}' | paste - - |awk '{print $1"\t" $2"\t" $4}' | tee -a result/$exe.data.txt
