exe = 	\
	1d-gs-3p\
	1d-gs-3p-blocking\
	1d-gs-lcs\
	1d-gs-lcs-blocking\
	2d-gs-5p\
	2d-gs-5p-blocking\
	3d-gs-7p\
	3d-gs-7p-blocking\
	1d-jacobi-3p\
	1d-jacobi-3p-blocking\
	2d-jacobi-5p\
	2d-jacobi-5p-blocking\
	2d-jacobi-9p\
	2d-jacobi-9p-blocking\
	2d-jacobi-life\
	2d-jacobi-life-blocking\
	3d-jacobi-7p\
	3d-jacobi-7p-blocking\
	3d-jacobi-27p\
	3d-jacobi-27p-blocking


all: 
	@-for d in $(exe); do \
		make -C $$d; \
		done
clean:
	@-for d in $(exe); do \
		make -C $$d clean; \
		done
