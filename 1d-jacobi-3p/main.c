#include "defines.h"

#define run_and_test(func, A)	for (x = 0; x < NX + 2 * XSTART; x++) {\
		A[0][x] = A_backup[0][x];\
		A[1][x] = A_backup[0][x];\
	}\
	gettimeofday(&start, 0);\
	func((double *)A, NX, T);\
	gettimeofday(&end, 0);\
	if(checkresult(NX, &A_correct[T % 2][0], &A[T % 2][0])) printf("Correct!\t");\
	printf(#func", NX = %d, T = %d, GStencil/s = %f\n", NX, T,\
			((double) NX * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L)


int main(int argc, char * argv[]) {

	struct timeval start, end;

	int x, t;

	int NX = atoi(argv[1]);
	int T = atoi(argv[2]);

	double (* A) [NX + 2 * XSTART]			= (double(*)[NX + 2 * XSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * 2);
	double (* A_backup) [NX + 2 * XSTART]	= (double(*)[NX + 2 * XSTART])malloc(sizeof(double) * (NX + 2 * XSTART));
	double (* A_correct)[NX + 2 * XSTART]	= (double(*)[NX + 2 * XSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * 2);


	srand(100); 
	for (x = 0; x < NX + 2 * XSTART; x++) {
		A_backup[0][x] = INIT;
	}

	run_and_test(naive_scalar, A_correct);

	run_and_test(naive_vector, A);

	run_and_test(vectime, A);


	free(A_correct);
	free(A);
	free(A_backup);

	return 0;
}



int checkresult(int NX, double * A_correct, double * A) {
	int correct = 1;
	int x;
#ifdef CHECK
	for (x = XSTART; x < NX + XSTART; x++) {
		if(A_correct[x] != A[x]){
			printf("x = [%d], Correct = %f, Wrong = %f\n", x, A_correct[x], A[x]);
			correct = 0;
		}
	}
#endif
	return correct;
}




