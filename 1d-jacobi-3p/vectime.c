#include "defines.h"

void vectime(double * A, int NX, int T){

	double (* B)[NX + 2 * XSTART] = (double(*)[NX + 2 * XSTART]) A;

	int x, t = 0, tt, xx;

	__m256d v0, v1, v2, v3, v4, v5, v6, v7;
	__m256d mid_in, mid_out;
	__m256d out, in;
	SET_COFF;

	for ( tt = 0; tt <= T - VECLEN; tt += VECLEN){	


		//  x 
		//	3|3333333
		//	        22222222**
		//	                 11111111
		//	                      00000000#|#
		//								  x + STRIDE * VECLEN
		for( t = tt ; t < tt + VECLEN - 1 ; t++){
			#pragma vector always
			#pragma ivdep
			for ( x = XSTART; x < XSTART + STRIDE * (VECLEN - 1 - (t - tt)) + ((t < tt + VECLEN / 2) ? LANESTRIDE : 0); x++){
				Compute_scalar(B, t, x);
			}
		}

		t = tt;

		x = XSTART - XSLOPE;

		vload(v3, B[(t + 1) % 2]	[x]);
		vload(v7, B[(t + 1) % 2]	[x + VECLEN]);
		vload(v2, B[(t) % 2]		[x + STRIDE]);
		vload(v6, B[(t) % 2]		[x + VECLEN + STRIDE]);
		vload(v1, B[(t + 1) % 2]	[x + 2 * STRIDE + LANESTRIDE]);
		vload(v5, B[(t + 1) % 2]	[x + VECLEN + 2 * STRIDE + LANESTRIDE]);
		vload(v0, B[(t) % 2]		[x + 3 * STRIDE + LANESTRIDE]);
		vload(v4, B[(t) % 2]		[x + VECLEN + 3 * STRIDE + LANESTRIDE]);

		transpose(v0, v1, v2, v3, out, in);
		transpose(v4, v5, v6, v7, out, in);

		vload(mid_in, B[(t) % 2][x + 2 * STRIDE - 1]);
		vload(in, B[(t) % 2][x + XSLOPE + STRIDE * VECLEN + LANESTRIDE]);

		mid_in = _mm256_blend_pd(in, mid_in, 0b1100);

		for ( x = XSTART; x < NX + XSTART - STRIDE * VECLEN - LANESTRIDE; x += STRIDE+XSLOPE){
			// Compute_4vector(v0,v1,v2,v3,v4,v5);	//-> v0~v3

			Compute_1vector(v0, v1, v2);
			Input_Output_double_strides_02(out, v0, mid_in);

			Compute_1vector(v1, v2, v3);
			Input_Output_double_strides_13(out, v1, mid_in);
			
			mid_in = _mm256_permute2f128_pd(out, in, 0x03);	

			Compute_1vector(v2, v3, v4);
			Input_Output_double_strides_02(mid_out, v2, mid_in);

			Compute_1vector(v3, v4, v5);
			Input_Output_double_strides_13(mid_out, v3, mid_in);

			out = _mm256_permute2f128_pd(out, mid_out, 0x31);
			_mm256_storeu_pd(&B[(t) % 2][x], out);	

			vload(in, B[(t) % 2][x + STRIDE * VECLEN + LANESTRIDE + 2 * 2]);

			// Compute_4vector(v4,v5,v6,v7,v0,v1);	//-> v4~v7		

			mid_in = _mm256_permute2f128_pd(mid_out, in, 0x02);

			Compute_1vector(v4, v5, v6);
			Input_Output_double_strides_02(out, v4, mid_in);

			Compute_1vector(v5, v6, v7);
			Input_Output_double_strides_13(out, v5, mid_in);

			mid_in = _mm256_permute2f128_pd(out, in, 0x03);

			Compute_1vector(v6, v7, v0);
			Input_Output_double_strides_02(mid_out, v6, mid_in);	


			Compute_1vector(v7, v0, v1);
			Input_Output_double_strides_13(mid_out, v7, mid_in);

			out  = _mm256_permute2f128_pd(out, mid_out, 0x31);
			_mm256_storeu_pd(&B[(t) % 2][x + VECLEN], out);	

			vload(in, B[(t) % 2][x + STRIDE * VECLEN + LANESTRIDE + 4*2]);

			mid_in = _mm256_permute2f128_pd(mid_out, in, 0x02);
		}

		vstore(B[(t) % 2][x-2+STRIDE*2], mid_in);

		transpose(v0, v1, v2, v3, out, in);
		transpose(v4, v5, v6, v7, out, in);
		
		vstore(B[(t + 1) % 2]	[x - 1], 									v3);
		vstore(B[(t + 1) % 2]	[x - 1 + VECLEN], 					 		v7);
		vstore(B[(t) % 2]		[x - 1 + STRIDE], 	 						v2);
		vstore(B[(t) % 2]		[x - 1 + VECLEN + STRIDE],  				v6);
		vstore(B[(t + 1) % 2]	[x - 1 + 2 * STRIDE + LANESTRIDE], 			v1);
		vstore(B[(t + 1) % 2]	[x - 1 + VECLEN + 2 * STRIDE + LANESTRIDE],	v5);	

		xx = x + STRIDE * (VECLEN - 1) + LANESTRIDE;

		for(t = tt ; t < tt + VECLEN ; t++, xx -= STRIDE){	
			if ((t - tt) == VECLEN / 2)	xx -= LANESTRIDE;		
			#pragma vector always
			#pragma ivdep
			for (x = xx; x < NX + XSTART; x ++){
				Compute_scalar(B, t, x);;
			}
		}
	}
	for ( ; t < T; t++){
		#pragma vector always
		#pragma ivdep
		for (x = XSTART; x < NX + XSTART; x++) {
			Compute_scalar(B, t, x);
		}
	}	
}
