#include "define.h"

int naive_scalar(char * x1, char * x2, int nx1, int nx2, int x1b, int x2b, int * lcs){

	int t, x, k;
	int lcs_t_minus_1_x_minus_1;
	int lcs_t_x;

    for (t = XSTART; t < nx1 + XSTART; t++) {
        lcs_t_minus_1_x_minus_1 = 0;

        for ( k = LCSSTART, x = XSTART; x < nx2 + XSTART; x++, k++) {
            if ( x1[t] == x2[x] ) lcs_t_x = 1 + lcs_t_minus_1_x_minus_1;
            else lcs_t_x = max(lcs[k-1], lcs[k]);
			lcs_t_minus_1_x_minus_1 = lcs[k];
			lcs[k] = lcs_t_x;
        }          
    }
	return lcs_t_x;
}
