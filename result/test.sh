#! /bin/bash

np=24;
npstride=4;

./1d-gs-3p.sh
./1d-gs-3p-blocking.sh $np $npstride

./1d-gs-lcs.sh
./1d-gs-lcs-blocking.sh $np $npstride

./2d-gs-5p.sh
./2d-gs-5p-blocking.sh $np $npstride

./3d-gs-7p.sh
./3d-gs-7p-blocking.sh $np $npstride

./1d-jacobi-3p.sh
./1d-jacobi-3p-blocking.sh $np $npstride

./2d-jacobi-5p.sh
./2d-jacobi-5p-blocking.sh $np $npstride

./2d-jacobi-9p.sh
./2d-jacobi-9p-blocking.sh $np $npstride

./2d-jacobi-life.sh
./2d-jacobi-life-blocking.sh $np $npstride

./3d-jacobi-7p.sh
./3d-jacobi-7p-blocking.sh $np $npstride

./3d-jacobi-27p.sh
./3d-jacobi-27p-blocking.sh $np $npstride




