#include "defines.h"

void naive_scalar(double * A, int N, int T, int xb, int tb){

	int x, t;
	for (t = 0; t < T; t++) {
		for (x = XSTART; x < N + XSTART; x++) {
			Compute_scalar(A,x);
		}
	}
}
