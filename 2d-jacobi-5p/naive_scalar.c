#include "define.h"

void naive_scalar(double * A, int NX, int NY, int T){
	int t, x, y;
	double (* A_correct)[NX + 2 * XSTART][ NY + 2 * YSTART] =  (double (*)[NX + 2 * XSTART][ NY + 2 * YSTART]) A;
	for (t = 0; t < T; t++) {
		for (x = XSTART; x < NX + XSTART; x++) {
			#pragma novector
			for (y = YSTART; y < NY + YSTART; y++) {
				Compute_scalar(A_correct, t, x, y);
			}
		}
	}
}
