#include "define.h"

void vectime(double * A, int N, int T){

	int x, t = 0, tt, xx;
	__m256d v0, v1, v2, v3, v4, v5, v6, v7, vleft;
	__m256d out, in, out2, in2;
	SET_COFF;

	if( N < STRIDE * (VECLEN + 1)){
		printf("The length is too small to call this vector code!\n");
		return;
	}

	for ( tt = 0; tt <= T - VECLEN; tt += VECLEN){	


		// 0|0000000000000000000000000000
		// 0|3333333322222222111111110000
		for( t = tt ; t < tt + VECLEN - 1 ; t++){
			for ( x = XSTART; x < XSTART + STRIDE * (VECLEN - 1 - (t - tt)) ; x++){
				Compute_scalar(A, x);	
			}
		}

		v3 = _mm256_loadu_pd(&A[XSTART + 0 * VECLEN]);
		v7 = _mm256_loadu_pd(&A[XSTART + 1 * VECLEN]);
		v2 = _mm256_loadu_pd(&A[XSTART + 2 * VECLEN]);
		v6 = _mm256_loadu_pd(&A[XSTART + 3 * VECLEN]);
		v1 = _mm256_loadu_pd(&A[XSTART + 4 * VECLEN]);
		v5 = _mm256_loadu_pd(&A[XSTART + 5 * VECLEN]);
		v0 = _mm256_loadu_pd(&A[XSTART + 6 * VECLEN]);
		v4 = _mm256_loadu_pd(&A[XSTART + 7 * VECLEN]);

		transpose(v0, v1, v2, v3, in, in2);
		transpose(v4, v5, v6, v7, in, in2);

		vleft = _mm256_set_pd(A[XSTART - XSLOPE + 0 * STRIDE], \
							  A[XSTART - XSLOPE + 1 * STRIDE], \
							  A[XSTART - XSLOPE + 2 * STRIDE], \
							  A[XSTART - XSLOPE + 3 * STRIDE]);  
/*
STRIDE = 8
abcdefgh
01234567abcdefgh
        01234567abcdefgh
                01234567abcdefgh
                        01234567
*/
		for ( x = XSTART; x <= N + XSTART - STRIDE * VECLEN - STRIDE - 1; x += STRIDE){	

			in	= _mm256_loadu_pd(&A[x + STRIDE * VECLEN]);	
			in2 = _mm256_loadu_pd(&A[x + STRIDE * VECLEN + VECLEN]);

			Compute_1vector(vleft, v0, v1);
			Compute_1vector(v0, v1, v2);
			Compute_1vector(v1, v2, v3);
			Compute_1vector(v2, v3, v4);
			Compute_1vector(v3, v4, v5);
			Compute_1vector(v4, v5, v6);
			Compute_1vector(v5, v6, v7);

			Input_Output_1(out, v0, in);
			Input_Output_2(out, v1, in);
			Input_Output_3(out, v2, in);
			Input_Output_4(out, v3, in);
			_mm256_storeu_pd(&A[x], out);

			Compute_1vector(v6, v7, v0);
			
			vleft = v7;

			Input_Output_1(out2, v4, in2);
			Input_Output_2(out2, v5, in2);
			Input_Output_3(out2, v6, in2);
			Input_Output_4(out2, v7, in2);
			_mm256_storeu_pd(&A[x + VECLEN], out2);			
						
		}

		in = _mm256_loadu_pd(&A[x + STRIDE * VECLEN]);

		Compute_1vector(vleft, v0, v1);
		Compute_1vector(v0, v1, v2);
		Compute_1vector(v1, v2, v3);
		Compute_1vector(v2, v3, v4);
		Compute_1vector(v3, v4, v5);
		Compute_1vector(v4, v5, v6);
		Compute_1vector(v5, v6, v7);

		out = vrotate_high2low(v0);
		out = _mm256_blend_pd(out, in, 0b0001);

		Compute_1vector(v6, v7, out);


		transpose(v0, v1, v2, v3, in, in2);
		transpose(v4, v5, v6, v7, in, in2);
		
		_mm256_storeu_pd(&A[x + 0 * VECLEN], v3);
		_mm256_storeu_pd(&A[x + 1 * VECLEN], v7);
		_mm256_storeu_pd(&A[x + 2 * VECLEN], v2);
		_mm256_storeu_pd(&A[x + 3 * VECLEN], v6);
		_mm256_storeu_pd(&A[x + 4 * VECLEN], v1);
		_mm256_storeu_pd(&A[x + 5 * VECLEN], v5);
		_mm256_storeu_pd(&A[x + 6 * VECLEN], v0);
		_mm256_storeu_pd(&A[x + 7 * VECLEN], v4);	

		xx = x + STRIDE * VECLEN;
		for(t = tt ; t < tt + VECLEN ; t++, xx -= STRIDE){	
			for ( x = xx; x < N + XSTART; x++){
				Compute_scalar(A, x);
			}
		}
	}

	for ( ; t < T; t++){
		for (x = XSTART; x < N + XSTART; x++) {
			Compute_scalar(A, x);
		}
	}	
}
