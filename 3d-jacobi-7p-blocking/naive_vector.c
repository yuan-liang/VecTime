#include "defines.h"

void naive_vec(double * A, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb) {
	double (* A_correct)[NX + 2 * XSTART][ NY + 2 * YSTART][ NZ + 2 * ZSTART] =  (double (*)[NX + 2 * XSTART][ NY + 2 * YSTART][ NZ + 2 * ZSTART]) A;
    int t, x, y, z;
	int xxx, yyy, zzz;
    for (t = 0; t < T; t++) {
		for (x = XSTART; x < NX + XSTART; x++) {
			for (y = YSTART; y < NY + YSTART; y++) {
            	#pragma ivdep
            	#pragma vector always
				for (z = ZSTART; z < NZ + ZSTART; z++) {
					Compute_scalar(A_correct,t,x,y,z);
				}
			}
		}		
	}
}