#include "defines.h"

#define run_and_test(func, A)	for (x = 0; x < NX + 2 * XSTART; x++) {\
		for (y = 0; y < NY + 2 * YSTART; y++) {\
			for (z = 0; z < NZ + 2 * ZSTART; z++) {\
				A[0][x][y][z] = A_backup[0][x][y][z];\
				A[1][x][y][z] = A_backup[0][x][y][z];\
			}\
		}\
	}\
	gettimeofday(&start, 0);\
	func((double *)A, NX, NY, NZ, T, xb, yb, zb, tb);\
	gettimeofday(&end, 0);\
	if(checkresult(NX, NY, NZ, ( double (* )[NY+2*YSTART][ NZ+2*ZSTART])&A_correct[T % 2][0][0][0], ( double (* )[NY+2*YSTART][ NZ+2*ZSTART])&A[T % 2][0][0][0])) printf("Correct!\t");\
	printf(#func", NX = %d, NY = %d, NZ = %d, T = %d, GStencil/s = %f\n",\
	 								NX, NY, NZ, T,\
									((double) NX * NY * NZ * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L)


int main(int argc, char* argv[]) {

	struct timeval start, end;
	long int i, j, k;
	int x, y, z, t;
	if (argc != 9) {
		printf("usage: %s <NX> <NY> <NZ> <T> <xb> <yb> <zb> <tb>\n", argv[0]);
		return 0;
	}
	int NX  = atoi(argv[1]);
	int NY  = atoi(argv[2]);
	int NZ  = atoi(argv[3]);
	int T   = atoi(argv[4]);
	int xb  = atoi(argv[5]);
	int yb  = atoi(argv[6]);
	int zb	= atoi(argv[7]);
	int tb  = atoi(argv[8]);
	
	double (*A)[NX + 2 * XSTART][NY+2*YSTART][NZ+2*ZSTART] = (double(*)[NX + 2 * XSTART][NY+2*YSTART][NZ+2*ZSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART) * (NZ + 2 * ZSTART) * 2);
	double (*A_correct)[NX + 2 * XSTART][NY+2*YSTART][NZ+2*ZSTART] = (double(*)[NX + 2 * XSTART][NY+2*YSTART][NZ+2*ZSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART) * (NZ + 2 * ZSTART) * 2);
	double (*A_backup)[NX + 2 * XSTART][NY+2*YSTART][NZ+2*ZSTART] = (double(*)[NX + 2 * XSTART][NY+2*YSTART][NZ+2*ZSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART) * (NZ + 2 * ZSTART) ) ;

	srand(100);
	for (i = 0; i < NX + 2 * XSTART; i++) {
		for (j = 0; j < NY + 2 * YSTART; j++) {
			for (k = 0; k < NZ + 2 * ZSTART; k++) {
				A_backup[0][i][j][k] = INIT;				
			}
		}
	}
#ifdef CHECK
	run_and_test(naive_vec, A_correct);
#endif
	run_and_test(vectime, A);


	return 0;
}

int checkresult( int NX, int NY, int NZ, double (* A_correct)[NY+2*YSTART][ NZ+2*ZSTART], double (* A)[NY+2*YSTART][ NZ+2*ZSTART] ) {
	int correct = 1;
#ifdef CHECK
	int x, y, z;
	for (x= XSTART; x < NX + XSTART; x++) {
		for (y = YSTART; y < NY + YSTART; y++) {
			for (z = ZSTART; z < NZ + ZSTART; z++) {
				if(A[x][y][z] != A_correct[x][y][z]){
					printf("x = [%d], y = [%d], z = [%d], Correct = %.15f, Wrong = %.15f\n", x-XSTART+XSLOPE, y-YSTART+YSLOPE, z, A_correct[x][y][z], A[x][y][z]);
					correct = 0;
				}
			}
		}
	}
#endif
	return correct;
}

void print256_vec(__m256d var, char str[] )
{
    double_t val[4];
    memcpy(val, &var, sizeof(val));
    printf("%s: %f %f %f %f \n", str, 
           val[3], val[2], val[1], val[0]);
}