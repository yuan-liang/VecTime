#include "define.h"
void naive_vector(double * A, int NX, int NY, int T, int xb, int yb, int tb){
	int t, x, y;
	int xxx,yyy;
	double (* A_correct)[NX + 2 * XSTART][ NY + 2 * YSTART] =  (double (*)[NX + 2 * XSTART][ NY + 2 * YSTART]) A;
	for (t = 0; t < T; t++) {
		for (x = XSTART; x < NX + XSTART; x++) {
			#pragma vector always
			#pragma ivdep
			for (y = YSTART; y < NY + YSTART; y++) {
				Compute_scalar(A_correct, t, x, y);				
			}
		}		
	}
}
