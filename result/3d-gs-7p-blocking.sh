#! /bin/bash



np=$1;
npstride=$2;
if  [[ $npstride != 1 ]]
then
    corenum[0]=1
    for i in `seq  $npstride $npstride $np`
    do
        corenum[$(($i/$npstride))]=$i;
        #echo ${corenum[$(($i/$npstride))]} $i
    done
else
    for i in `seq  1 $np`
    do
        corenum[$(($i-1))]=$i;
        #echo ${corenum[$(($i/$npstride))]} $i
    done
fi

total=16;
a=8;
b=8;

for exe in 3d-gs-7p-blocking
do
    rm  -f result/$exe.log
    for index in `seq $a $b`;
    do 
    nx=$((2 ** $index))
    t=$((2 ** ($total - $index)))
    for i in "${corenum[@]}";
    do
	export OMP_NUM_THREADS=$i;../$exe/exe-$exe 800 800 800 200 32 32 32 32 | tee -a result/$exe.log 
    	echo  $i >> result/$exe.cpulist
    done
    done
done

echo -e "np\tour" | tee result/$exe.data.txt
cat result/$exe.log|awk -F "= " '{print $NF}'|paste  result/$exe.cpulist -  | awk '{print $0}' | tee -a  result/$exe.data.txt
rm  result/$exe.cpulist



