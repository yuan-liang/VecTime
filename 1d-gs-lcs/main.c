#include "define.h"

#define run_and_test(func, lcs2)	for (x = LCSSTART - 1; x < NX2 + LCSSTART; x++) lcs2[x] = 0;\
	gettimeofday(&start, 0);\
	func(X1, X2, NX1, NX2, lcs2);\
	gettimeofday(&end, 0);\
	if(checkresult(NX2, lcs_correct, lcs2)) printf("Correct!\t");\
	printf(#func", NX1 = %d, NX2 = %d, lcs = %d, GStencil/s = %f\n", NX1, NX2, lcs2[LCSSTART + NX2 - 1],\
			((double) NX1 * NX2) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L)



int main(int argc, char * argv[]) {

	struct timeval start, end;

	int x, t;
	int NX1 = atoi(argv[1]);
	int NX2 = atoi(argv[2]);
	int T = NX1;

	char * X1 = (char*)malloc(sizeof(char) * (NX1 + 2 * XSTART));
	char * X2 = (char*)malloc(sizeof(char) * (NX2 + 2 * XSTART));
	int * lcs = (int*)malloc(sizeof(int) * (NX2 + 1 + 2 * LCSSTART));
#ifdef CHECK
	int * lcs_correct = (int*)malloc(sizeof(int) * (NX2 + 1 + 2 * LCSSTART));
#else 
	int * lcs_correct = lcs;
#endif

	int correct_result, result;

	srand(124); 
	for (x = 0; x < NX1 + 2 * XSTART; x++) {
		X1[x] = 'A' + (rand() % 26);
	}
	for (x = 0; x < NX2 + 2 * XSTART; x++) {
		X2[x] = 'A' + (rand() % 26);
	}
#ifdef CHECK
	run_and_test(naive_scalar, lcs_correct);
#endif

	run_and_test(vectime, lcs);

	free(X1);
	free(X2);
	free(lcs);

	return 0;
}

int checkresult(int N, int * A_correct, int * A) {
	int correct = 1;
	int x;
	for (x = LCSSTART; x < N + LCSSTART; x++) {
		if(A_correct[x] != A[x]){
			printf("x = [%d], Correct = %d, Wrong = %d\n", x, A_correct[x], A[x]);
			correct = 0;
		}
	}
	return correct;
}
