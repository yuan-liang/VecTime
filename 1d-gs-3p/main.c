#include "define.h"

#define run_and_test(func, A) 	for (x = 0; x < N + 2 * XSTART; x++) {\
		A[x] = A_backup[x];\
	}\
	gettimeofday(&start, 0);\
	func(A, N, T);\
	gettimeofday(&end, 0);\
	if(checkresult(N,A_correct, A))\
		printf("Correct!\t");\
	printf(#func", N = %d, T = %d, GStencil/s = %f\n", N, T,\
			((double) N * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);


int main(int argc, char * argv[]) {

	struct timeval start, end;

	int x, t;
	int N = atoi(argv[1]);
	int T = atoi(argv[2]);


	double * A = (double*)malloc(sizeof(double) * (N + 2 * XSTART));
	double * A_backup = (double*)malloc(sizeof(double) * (N + 2 * XSTART));
#ifdef CHECK	
	double * A_correct = (double*)malloc(sizeof(double) * (N + 2 * XSTART));
#else
	double * A_correct = A;
#endif

	srand(100); 
	for (x = 0; x < N + 2 * XSTART; x++) {
		A_backup[x] = INIT;
		A_correct[x] = A_backup[x];
	}

#ifdef CHECK
	run_and_test(naive_scalar, A_correct);
#endif

	run_and_test(vectime, A);

#ifdef CHECK
	free(A_correct);
#endif
	free(A);
	free(A_backup);

	return 0;
}





int checkresult(int N, double * A_correct, double * A) {
	int correct = 1;
	int x;
#ifdef CHECK
	for (x = XSTART; x < N + XSTART; x++) {
		if(A_correct[x] != A[x]){
			printf("x = [%d], Correct = %f, Wrong = %f\n", x, A_correct[x], A[x]);
			correct = 0;
		}
	}
#endif
	return correct;
}

