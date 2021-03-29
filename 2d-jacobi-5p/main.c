#include "define.h"

#define run_and_test(func, A)	for (x = 0; x < NX + 2 * XSTART; x++) {\
		for (y = 0; y < NY + 2 * YSTART; y++) {\
		A[0][x][y] = A_backup[0][x][y];\
		A[1][x][y] = A_backup[0][x][y];\
		}\
	}\
	gettimeofday(&start, 0);\
	func((double *)A, NX, NY, T);\
	gettimeofday(&end, 0);\
	if(checkresult(NX, NY, ( double (* )[ NY+2*YSTART])&A_correct[T % 2][0][0], ( double (* )[ NY+2*YSTART])&A[T % 2][0][0])) printf("Correct!\t");\
	printf(#func", NX = %d, NY = %d, T = %d, GStencil/s = %f\n", NX, NY, T,\
			((double) NX * NY * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L)


int main(int argc, char* argv[]) {

	struct timeval start, end;

	int x, y, t;
	int NX  = atoi(argv[1]);
	int NY  = atoi(argv[2]);
	int T   = atoi(argv[3]);


	double (*A)[NX + 2 * XSTART][NY+2*YSTART] = (double(*)[NX + 2 * XSTART][NY+2*YSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART) * 2);
#ifdef CHECK
	double (*A_correct)[NX + 2 * XSTART][NY+2*YSTART] = (double(*)[NX + 2 * XSTART][NY+2*YSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART) * 2);
#else
	double (*A_correct)[NX + 2 * XSTART][NY+2*YSTART] = A;
#endif
	double (*A_backup)[NX + 2 * XSTART][NY+2*YSTART] = (double(*)[NX + 2 * XSTART][NY+2*YSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART));



	srand(100); 
	for (x = 0; x < NX + 2 * XSTART; x++) {
		for (y = 0; y < NY + 2 * YSTART; y++) {
			A_backup[0][x][y] = INIT;
		}
	}


	run_and_test(naive_scalar, A_correct);

	run_and_test(naive_vector, A);

	run_and_test(vectime, A);

#ifdef CHECK
	free(A_correct);
#endif
	free(A);
	free(A_backup);
	return 0;
}

int checkresult( int NX, int NY, double (* A_correct)[ NY+2*YSTART], double (* A)[ NY+2*YSTART] ) {
	int correct = 1;
	int x, y;
#ifdef CHECK
	for (x= 0; x < NX + 2 * XSTART; x++) {
		for (y = 0; y < NY + 2 * YSTART; y++) {
			if(A[x][y] != A_correct[x][y]){
				printf("x = [%d], y = [%d], Correct = %f, Wrong = %f\n", x, y, A_correct[x][y], A[x][y]);
				correct = 0;
			}
		}
	}
#endif
	return correct;
}
