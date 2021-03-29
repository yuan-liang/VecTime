#include "define.h"

#define run_and_test(func, A)	for (x = 0; x < NX + 2 * XSTART; x++) {\
		A[0][x] = A_backup[0][x];\
		A[1][x] = A_backup[0][x];\
	}\
	gettimeofday(&start, 0);\
	func((double *)A, NX, T, xb, tb);\
	gettimeofday(&end, 0);\
	if(checkresult(NX, &A_correct[T % 2][0], &A[T % 2][0])) printf("Correct!\t");\
	printf(#func", NX = %d, T = %d, GStencil/s = %f\n", NX, T,\
			((double) NX * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L)


int main(int argc, char * argv[]) {

	struct timeval start, end;

	int x, t;

	int NX = atoi(argv[1]);
	int T = atoi(argv[2]);
	int xb = atoi(argv[3]);
	int tb = atoi(argv[4]);

	double (* A) [NX + 2 * XSTART]			= (double(*)[NX + 2 * XSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * 2);
	double (* A_backup) [NX + 2 * XSTART]	= (double(*)[NX + 2 * XSTART])malloc(sizeof(double) * (NX + 2 * XSTART));
#ifdef CHECK
	double (* A_correct)[NX + 2 * XSTART]	= (double(*)[NX + 2 * XSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * 2);
#else
	double (* A_correct)[NX + 2 * XSTART]	= A;
#endif

	srand(100); 
	for (x = 0; x < NX + 2 * XSTART; x++) {
		A_backup[0][x] = INIT;
	}

#ifdef CHECK
	run_and_test(naive_vector, A_correct);
#endif



	run_and_test(vectime, A);



#ifdef CHECK
	free(A_correct);
#endif

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




