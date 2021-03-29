#include "defines.h"

void naive_vector(double * A, int NX, int T){
	double (* B)[NX + 2 * XSTART] = (double(*)[NX + 2 * XSTART]) A;
	int x, t;
	for (t = 0; t < T; t++){
		#pragma vector always
		#pragma ivdep
		for (x = XSTART; x < NX + XSTART; x++) {
			Compute_scalar(B, t, x);
		}
	}	
}
