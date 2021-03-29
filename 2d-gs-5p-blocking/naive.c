#include "defines.h"

void naive(double * B, int NX, int NY, int T, int xb, int yb, int tb){

	int t, x, y;
	double (* A)[ NY+2*YSTART] =  (double (*)[ NY+2*YSTART]) B;
	for (t = 0; t < T; t++) {
		for (x = XSTART; x < NX + XSTART; x++) {
			for (y = YSTART; y < NY + YSTART; y++) {
				Compute_scalar(A,x,y);
			}
		}
	}
}
