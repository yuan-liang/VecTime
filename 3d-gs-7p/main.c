#include "defines.h"

int main(int argc, char * argv[]) {

	struct timeval start, end;

	int  x, y, z, t;
	int NX  = atoi(argv[1]);
	int NY  = atoi(argv[2]);
	int NZ  = atoi(argv[3]);
	int T   = atoi(argv[4]);


	double (*A)[NY+2*YSTART][NZ+2*ZSTART] = (double(*)[NY+2*YSTART][NZ+2*ZSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART) * (NZ + 2 * ZSTART));
#ifdef CHECK
	double (*A_correct)[NY+2*YSTART][NZ+2*ZSTART] = (double(*)[NY+2*YSTART][NZ+2*ZSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART) * (NZ + 2 * ZSTART));
#else
	double (*A_correct)[NY+2*YSTART][NZ+2*ZSTART] = A;
#endif
	double (*A_backup)[NY+2*YSTART][NZ+2*ZSTART] = (double(*)[NY+2*YSTART][NZ+2*ZSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * (NY + 2 * YSTART) * (NZ + 2 * ZSTART));



	srand(100); 
	for (x = 0; x < NX + 2 * XSTART; x++) {
		for (y = 0; y < NY + 2 * YSTART; y++) {
			for (z = 0; z < NZ + 2 * ZSTART; z++){
				A_backup[x][y][z] = INIT;
				A_correct[x][y][z] = A_backup[x][y][z];
			}
		}
	}

#ifdef CHECK
	gettimeofday(&start, 0);
	for (t = 0; t < T; t++) {
		for (x = XSTART; x < NX + XSTART; x++) {
			for (y = YSTART; y < NY + YSTART; y++) {
				for (z = ZSTART; z < NZ + ZSTART; z++){
					Compute_scalar(A_correct,x,y,z);
				}
			}
		}
	}

	gettimeofday(&end, 0);
	printf("Correct!\t");
	printf("Naive, NX = %d, NY = %d, NZ = %d, T = %d, GStencil/s = %f\n", NX,NY,NZ,T,\
			((double) NX * NY * NZ * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);

#endif

	for (x = 0; x < NX + 2 * XSTART; x++) {
		for (y = 0; y < NY + 2 * YSTART; y++) {
			for (z = 0; z < NZ + 2 * ZSTART; z++){
				A[x][y][z] = A_backup[x][y][z];
			}
		}
	}
	gettimeofday(&start, 0);
	vectime((double *)A, NX, NY, NZ, T);
	gettimeofday(&end, 0);
	if(checkresult(NX, NY, NZ, A_correct, A))
		printf("Correct!\t");
	printf("Vectime, NX = %d, NY = %d, NZ = %d, T = %d, GStencil/s = %f\n", NX,NY,NZ,T,\
			((double) NX * NY * NZ * T) / (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1.0e-6) / 1000000000L);



	free(A);
#ifdef CHECK
	free(A_correct);
#endif
	free(A_backup);

	return 0;
}





int checkresult( int NX, int NY, int NZ, double (* A_correct)[ NY+2*YSTART][NZ+2*ZSTART], double (* A)[ NY+2*YSTART][NZ+2*ZSTART] ) {
	int correct = 1;
	int x, y, z;
#ifdef CHECK
	for (x= XSTART; x < NX + XSTART; x++) {
		for (y = YSTART; y < NY + YSTART; y++) {
			for (z = ZSTART; z < NZ + ZSTART; z++){
				if(A[x][y][z] != A_correct[x][y][z]){
					printf("x = [%d], y = [%d], z = [%d], Correct = %f, Wrong = %f\n", x, y, z, A_correct[x][y][z], A[x][y][z]);
					correct = 0;
				}
			}
		}
	}
#endif
	return correct;
}