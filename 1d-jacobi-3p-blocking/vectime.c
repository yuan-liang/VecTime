#include "define.h"


void vectime(double* A, int N, int T, int Bx, int tb){

	double (* B)[N + 2 * XSTART] = (double(*)[N + 2 * XSTART]) A;

	int x, t=0, tv, tt, xx;
	tb -= tb % 4;	
	int bx = Bx - 2 * tb * XSLOPE;
	int ix = Bx + bx;   // ix is even
	int nb0[2] = { myfloor(N-Bx,ix), myfloor(N-Bx,ix) + 1 };	
	int nrestpoints = N % ix;
	int bx_first_B1 = (Bx + nrestpoints)/2;
	int bx_last_B1  = (Bx + nrestpoints) - bx_first_B1;
	int xright[2] = {bx_first_B1 + Bx + XSTART,  bx_first_B1 + (Bx - bx)/2 + XSTART};
	int level = 0;
	register int xmin, xmax;


	for (tt = -tb; tt < T ; tt += tb ){

		#pragma omp parallel for schedule(dynamic) private(tv, t, xmin, xmax, x)
		
		for(xx = 0; xx < nb0[level]; xx++) {
		
			__m256d v0,v1,v2,v3,v4,v5,v6,v7;
			__m256d mid_in, mid_out, in;
			__m256d out;
			SET_COFF;

			for(tv = max(tt, 0) ; tv <= min(tt + 2 * tb, T) - VECLEN; tv += VECLEN){

				for (t = tv; t < tv + VECLEN - 1; t++){
					xmin = (level == 1 && xx == 0) ? XSTART : (xright[level] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
					xmax = xmin + STRIDE * (VECLEN - 1 - (t - tv));
					if(!(level == 1 && xx == 0)){
						xmax += ((tt + tb < tv + 1) ? 1 : -1) * (VECLEN - 1 - (t - tv));
					}
					#pragma vector always
					#pragma ivdep
					for ( x = xmin; x < xmax; x++){
						Compute_scalar(B, t, x);					
					}
				}
				t = tv;

				x = (level == 1 && xx == 0) ? xmin-XSLOPE : (tt+tb < t+1 ? xmin : xmin-2*XSLOPE);
				xmax = (level == 1 && xx == nb0[1] -1) ? N + XSTART : (xright[level]      + xx*ix - myabs((tt+tb),(t+1))*XSLOPE);

				vload(v3, B[(t + 1) % 2]	[x]);
				vload(v7, B[(t + 1) % 2]	[x + VECLEN]);
				vload(v2, B[(t) % 2]		[x + STRIDE]);
				vload(v6, B[(t) % 2]		[x + VECLEN + STRIDE]);
				vload(v1, B[(t + 1) % 2]	[x + 2 * STRIDE]);
				vload(v5, B[(t + 1) % 2]	[x + VECLEN + 2 * STRIDE]);
				vload(v0, B[(t) % 2]		[x + 3 * STRIDE]);
				vload(v4, B[(t) % 2]		[x + VECLEN + 3 * STRIDE]);

				transpose(v0, v1, v2, v3, out, in);
				transpose(v4, v5, v6, v7, out, in);


				vload(in, B[(t) % 2][x + XSLOPE + STRIDE * VECLEN]);


				for ( x += XSLOPE; x < xmax - STRIDE * VECLEN; x += STRIDE+XSLOPE){				
					// Compute_4vector(v0,v1,v2,v3,v4,v5);	//-> v0~v3


					Compute_4vector(v0,v1,v2,v3,v4,v5);	//-> v0~v3

					Input_Output_1(out, v0, in);
					Input_Output_2(out, v1, in);
					Input_Output_3(out, v2, in);
					Input_Output_4(out, v3, in);


					_mm256_storeu_pd(&B[(t) % 2][x], out);	
					vload(in, B[(t) % 2][x + STRIDE * VECLEN + VECLEN]);

					Compute_4vector(v4,v5,v6,v7,v0,v1);	//-> v4~v7		

					Input_Output_1(out, v4, in);
					Input_Output_2(out, v5, in);
					Input_Output_3(out, v6, in);
					Input_Output_4(out, v7, in);			
					_mm256_storeu_pd(&B[(t) % 2][x + VECLEN], out);	
					vload(in, B[(t) % 2][x + STRIDE * VECLEN + VECLEN * 2]);	
				}

				transpose(v0, v1, v2, v3, out, in);
				transpose(v4, v5, v6, v7, out, in);
				
				vstore(B[(t + 1) % 2]	[x - 1], 									v3);
				vstore(B[(t + 1) % 2]	[x - 1 + VECLEN], 					 		v7);
				vstore(B[(t) % 2]		[x - 1 + STRIDE], 	 						v2);
				vstore(B[(t) % 2]		[x - 1 + VECLEN + STRIDE],  				v6);
				vstore(B[(t + 1) % 2]	[x - 1 + 2 * STRIDE], 			v1);
				vstore(B[(t + 1) % 2]	[x - 1 + VECLEN + 2 * STRIDE],	v5);

				xmin = x + STRIDE * (VECLEN - 1);

				for(t = tv ; t < tv + VECLEN ; t++, xmin -= STRIDE){		
					xmax = (level == 1 && xx == nb0[1] - 1) ? N + XSTART : (xright[level] + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);
					#pragma vector always
					#pragma ivdep					
					for (x = xmin; x < xmax; x++){
						Compute_scalar(B, t, x);
					}
				}
			}
		}
		level = 1 - level;
	}
	for (t = tt; t < T; t++){
		#pragma vector always
		#pragma ivdep
		for (x = XSTART; x < N + XSTART; x++) {
			Compute_scalar(B, t, x);
		}
	}
}
