#### Overview

This package includes the temporal vectorization codes for a set of Jacobi and Gauss-Seidel stencils.
In particular, 6 Jacobi: Heat-1D, -2D, -3D, Laplacian-2D, 3D and
a particular variant called B2S23 of Conway’s Game of Life 
used in Pluto, and 4 Gauss-Seidel stencils: GS-1D, -2D, -3D 
and the Longest common subsequence (LCS).
Each folder contains a sequential or parallel (with "-blocking" suffix) implementation for one stencil.



#### DEPENDENCIES

We tested the package on CentOS release 8.2.2004 using a machine with two Intel(R) Xeon(R) CPUs  (E5-2670 v3 @ 2.30GHz) with ICC version 19.1.1.217.
GCC also tested on this machine.
ICC or GCC compiler with OpenMP is sufficient to run the packge.


#### LICENSE

This package is available under GPL v3.


#### INSTALLING

* Step1: Simply run 'make' in the main directory. It will generate an executable in each stencil folder.

* Step2: modify the problem sizes and blocking sizes and run './test.sh'. Or if you want to test each kernel, goto the specific folder. For sequential test of a d-dimensional stencil, run ./exe $N1 ... $Nd $NT, where $Ni is the size of the i-th space dimension and
    $NT is the size of time dimension. For parallel test, run ./exe $N1 ... $Nd $NT $B1 ... $Bd $BT, where $Bi is the blocking size of the i-th space dimension and
    $BT is the blocking size of time dimension.
    The output is performance measured by GStenils/s.
