#include "defines.h"

#define run_and_test(func, B) 	for (x = 0; x < NX + 2 * XSTART; x++) {\
		for (y = 0; y < NY + 2 * YSTART; y++) {\
			B[x][y] =A_backup[x][y];\
		}\
	}\
	gettimeofday(&start, 0);\
	func((double *)B, NX, NY, T, xb, yb, tb);\
	gettimeofday(&end, 0);\
	if(checkresult(NX, NY,A_correct, B))\
		printf("Correct!\t");\
	printf(#func", NX = %d, NY = %d, T = %d, GStencil/s = %f\n", NX,NY,T,\
			((double) NX * NY * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);



int main(int argc, char * argv[]) {

	struct timeval start, end;

	int  x, y, t;

	int NX  = atoi(argv[1]);
	int NY  = atoi(argv[2]);
	int T   = atoi(argv[3]);
	int xb  = atoi(argv[4]);
	int yb  = atoi(argv[5]);
	int tb  = atoi(argv[6]);


	double (*A_backup)[NY+2*YSTART] = (double(*)[NY+2*YSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART));
	double (*A)[NY+2*YSTART] = (double(*)[NY+2*YSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART));
#ifdef CHECK
	double (*A_correct)[NY+2*YSTART] = (double(*)[NY+2*YSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART));
#else
	double (*A_correct)[NY+2*YSTART] = A;
#endif


	srand(100); 
	for (x = 0; x < NX + 2 * XSTART; x++) {
		for (y = 0; y < NY + 2 * YSTART; y++) {
			A_backup[x][y] = INIT;
		}
	}


#ifdef CHECK
	run_and_test(naive, A_correct);

#endif

	run_and_test(vectime, A);

#ifdef CHECK
	free(A_correct);
#endif
	free(A_backup);
	free(A);
	return 0;
}





int checkresult( int NX, int NY, double (* A)[ NY+2*YSTART], double (* B)[ NY+2*YSTART] ) {
	int correct = 1;
	int x, y;
#ifdef CHECK
	for (x= XSTART; x < NX + XSTART; x++) {
		for (y = YSTART; y < NY + YSTART; y++) {
			if(A[x][y] != B[x][y]){
				printf("x = [%d], y = [%d], Correct = %f, Wrong = %f\n", x, y, A[x][y], B[x][y]);
				correct = 0;
			}
		}
	}
#endif
	return correct;
}