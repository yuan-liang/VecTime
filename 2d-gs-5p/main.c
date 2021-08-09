#include "define.h"


#define run_and_test(func, A)	for (x = 0; x < NX + 2 * XSTART; x++) {\
		for (y = 0; y < NY + 2 * YSTART; y++) {\
			A[x][y] = A_backup[x][y];\
		}\
	}\
	gettimeofday(&start, 0);\
	func((double *)A, NX, NY, T);\
	gettimeofday(&end, 0);\
	if(checkresult(NX, NY, ( double (* )[ NY+2*YSTART])&A_correct[0][0], ( double (* )[ NY+2*YSTART])&A[0][0])) printf("Correct!\t");\
	printf(#func", NX = %d, NY = %d, T = %d, GStencil/s = %f\n", NX, NY, T,\
			((double) NX * NY * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L)


int main(int argc, char * argv[]) {

	struct timeval start, end;

	int x, y, t;
	int NX  = atoi(argv[1]);
	int NY  = atoi(argv[2]);
	int T   = atoi(argv[3]);


	double (*A)[NY+2*YSTART] = (double(*)[NY+2*YSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART));
#ifdef CHECK
	double (*A_correct)[NY+2*YSTART] = (double(*)[NY+2*YSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART));
#else
	double (* A_correct) [NY+2*YSTART] = A;
#endif
	double (*A_backup)[NY+2*YSTART] = (double(*)[NY+2*YSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART));



	srand(100); 
	for (x = 0; x < NX + 2 * XSTART; x++) {
		for (y = 0; y < NY + 2 * YSTART; y++) {
			A_backup[x][y] = 1.0 * (rand() % 1024);
		}
	}

//#ifdef CHECK
	run_and_test(scalar, A_correct);
//#endif
	run_and_test(vectime, A);


	free(A);
#ifdef CHECK
	free(A_correct);
#endif
	free(A_backup);

	return 0;
}



int checkresult( int NX, int NY, double (* A_correct)[ NY+2*YSTART], double (* A)[ NY+2*YSTART] ) {
	int correct = 1;
	int x, y;
#ifdef CHECK
	for (x= XSTART; x < NX + XSTART; x++) {
		for (y = YSTART; y < NY + YSTART; y++) {
			if(A[x][y] != A_correct[x][y]){
				printf("x = [%d], y = [%d], Correct = %f, Wrong = %f\n", x, y, A_correct[x][y], A[x][y]);
				correct = 0;
			}
		}
	}
#endif
	return correct;
}
