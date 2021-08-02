
for exe in 1d*p 1d*lcs
do
    echo $exe
    ./$exe/exe-$exe 1024 1000
done

for exe in 1d*p-blocking 1d*lcs-blocking
do
    echo $exe
    export OMP_NUM_THREADS=12
    ./$exe/exe-$exe 160000 1000 1024 128
done


for exe in 2d*p 2d*life
do
    echo $exe
    ./$exe/exe-$exe 256 256 1000
done

for exe in 2d*p-blocking 2d*life-blocking
do
    echo $exe
    export OMP_NUM_THREADS=12
    ./$exe/exe-$exe 2000 2000 100 64 64 16
done

for exe in 3d*p 
do
    echo $exe
    ./$exe/exe-$exe 64 64 64 128
done

for exe in 3d*p-blocking 
do
    echo $exe
    export OMP_NUM_THREADS=12
    ./$exe/exe-$exe 200 200 200 100 32 32 32 8
done

