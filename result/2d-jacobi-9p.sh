#! /bin/bash




total=24;
a=7;
b=13;

for exe in  2d-jacobi-9p 
do
    rm  -f result/$exe.log
    for index in `seq $a $b`;
    do 
    nx=$((2 ** $index))
    t=$((2 ** (total - $a - ($index - $a) * 2)))
    ../$exe/exe-$exe $nx $nx $t  |grep GS | tee -a result/$exe.log
    done
done
echo -e "size\tscalar\tauto\tour" | tee result/$exe.data.txt
cat result/$exe.log |tr "," "=" | awk -F "= " '{print $3 "\t" $NF}'|paste - - - |awk '{print $1"\t" $2"\t" $4"\t" $6}' | tee -a result/$exe.data.txt
