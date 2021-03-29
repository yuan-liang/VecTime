#include "defines.h"

void vectime(double * B, int NX, int NY, int T, int xb, int yb, int tb){

	double (* A)[ NY + 2 * YSTART] =  (double (*)[ NY + 2 * YSTART])  B;

	int wave;
	int tt, xx, yy;
	int t, x, y;
	int xbeg, ybeg;
	int myxb, myyb;
	int myxbeg, myybeg;
#ifdef scalarratio
	long long int cnt = 0;
#endif

	const int xblocknum = myceil(NX + T - 1, xb);
	const int yblocknum = myceil(NY + T - 1, yb);

	for (wave = 0; wave < myceil(T, tb) + myceil(NX + T - 1, xb) - 1 + myceil(NY + T - 1, yb) - 1; wave++) {

		#pragma omp parallel for private(tt, xx, yy, t, x, y, xbeg, ybeg, myxb, myyb, myxbeg, myybeg) collapse(2) schedule(dynamic, 1)

		for (xx = 0; xx < xblocknum; xx++){

			for (yy = 0; yy < yblocknum; yy++){	
		
				__m256d v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11;
				__m256d out, in_0, in_4;
				SET_COFF;
				double tmp[4];

				xbeg = XSTART + xx * xb - (wave - xx - yy) * tb;
				ybeg = YSTART + yy * yb - (wave - xx - yy) * tb;

				for ( tt = max(0, (wave - xx - yy) * tb); tt <= min(T, (wave - xx - yy + 1) * tb) - VECLEN; tt += VECLEN, xbeg -= VECLEN, ybeg -= VECLEN ){

					if(xbeg - VECLEN + 1 < XSTART){
						myxbeg 	= 	XSTART + VECLEN - 1;
						myxb	=	xb - (myxbeg - xbeg);
					} else {
						myxbeg	=	xbeg;
						myxb	=	xb;
					}
					if (xbeg + xb >= XSTART + NX){	
						myxb -= xbeg + xb - (XSTART + NX);
					}

					if(ybeg - VECLEN + 1 < YSTART){
						myybeg 	= 	YSTART + VECLEN - 1;
						myyb	=	yb - (myybeg - ybeg);
					} else {
						myybeg	=	ybeg;
						myyb	=	yb;
					}
					if (ybeg + yb >= YSTART + NY){	
						myyb -= ybeg + yb - (YSTART + NY);
					}

					if (myxb <=  VECLEN || myyb <= 2 * VECLEN){

						for (t = tt; t < tt + VECLEN; t++){
							for (x = max(XSTART, xbeg - (t - tt)); x < min(NX + XSTART, xbeg - (t - tt) + xb); x++) {
								for (y = max(YSTART, ybeg - (t - tt)); y < min(NY + YSTART, ybeg - (t - tt) + yb); y++) {
									Compute_scalar(A, x, y);

								}
							}
						}	

					} else {

						for( t = tt ; t < tt + VECLEN - 1 ; t++){
							//		for myxbeg != xbeg
							for ( x = max(XSTART, xbeg - (t - tt)); x < myxbeg - (t - tt); x++){
								for ( y = max(YSTART, ybeg - (t - tt)); y < myybeg - (t - tt) + myyb ; y++){
									Compute_scalar(A, x, y);	
								}
							}

							//		for myybeg != ybeg
							for(x = myxbeg - (t - tt); x <min(NX + XSTART, xbeg + xb - (t - tt)); x++){
								for ( y =max(YSTART, ybeg - (t - tt)); y < myybeg - (t - tt); y++){									
									Compute_scalar(A, x, y);
								}
							}
						}

						//			 * <--> xbeg
						//		4|32100000|
						//		4|33221100|
						//		to enter the vector compute code, xb should be at least VECLEN
						//		xbeg - VECLEN + 1 >= XSTART
						//		xb > VECLEN
						for( t = tt ; t < tt + VECLEN - 1 ; t++){
							for ( x = myxbeg - (t - tt); x < myxbeg + VECLEN - 1 - STRIDE * (t - tt); x++){
								for ( y = myybeg - (t - tt); y < myybeg - (t - tt) + myyb ; y++){
									Compute_scalar(A, x, y);	
								}
							}
						}
						//		3 2 1 0  --- transpose
						//		 3 2 1 0 --- transpose
						//		y is also shifted
						for(x = myxbeg + VECLEN - 1; x < myxbeg + VECLEN - 1 + STRIDE; x++){
							for ( y = myybeg - VECLEN + 1; y <= myybeg - VECLEN + 1 + myyb - VECLEN; y += VECLEN ){
								v1 = _mm256_loadu_pd(&A[x]					[y + 3]);
								v2 = _mm256_loadu_pd(&A[x  - STRIDE]		[y + 2]);
								v3 = _mm256_loadu_pd(&A[x  - STRIDE * 2]	[y + 1]);
								v4 = _mm256_loadu_pd(&A[x  - STRIDE * 3]	[y]);
								transpose(v1, v2, v3, v4, v5, v6);
								_mm256_storeu_pd(&A[x]					[y + 3], v1);
								_mm256_storeu_pd(&A[x  - STRIDE]		[y + 2], v2);
								_mm256_storeu_pd(&A[x  - STRIDE * 2]	[y + 1], v3);
								_mm256_storeu_pd(&A[x  - STRIDE * 3]	[y], v4);
							}
						}
		
						//	xbeg + xb should not be transposed
						for(x = myxbeg + VECLEN + 1; x < myxbeg + myxb; x++){

							y = myybeg - VECLEN + 1;

							v1 = _mm256_set_pd( A[x - 2 - STRIDE * 3]	 [y - 1], \
												A[x - 2 - STRIDE * 2]	 [y], \
												A[x - 2 - STRIDE]	 	 [y + 1], \
												A[x - 2]			 	 [y + 2]); 

							v0 = _mm256_set_pd( A[x - STRIDE * 3]	 [y], \
												A[x - STRIDE * 2]	 [y + 1], \
												A[x - STRIDE]	 	 [y + 2], \
												A[x]			 	 [y + 3]); 						

							in_4 = _mm256_loadu_pd(&A[x - 1 - STRIDE * VECLEN][y]);

							v5 = _mm256_loadu_pd(&A[x - 1][y + 3]);
							v8 = _mm256_loadu_pd(&A[x - 1 - STRIDE][y + 2]);
							v4 = _mm256_loadu_pd(&A[x - 2][y + 3]);

							Input_high_1(v3, v8, in_4);

							for ( ; y <= myybeg - VECLEN + 1 + myyb - VECLEN; y += VECLEN){	

								in_0 = _mm256_loadu_pd(&A[x][y + 4]);
								_mm256_storeu_pd(&A[x][y + 3], v0);

								v11 = _mm256_loadu_pd(&A[x - 1 - STRIDE * 2][y + 1]);
								v7 = _mm256_loadu_pd(&A[x - 2 - STRIDE][y + 2]);
								Input_high_2(v6, v11, in_4);
								Compute_1vector(    v3, \
												v1, v4, v7, \
													v5);
								Input_Output_keep_src_1(out,v0,v4,in_0);
								_mm256_storeu_pd(&A[x - STRIDE][y + 2], v0);



								v2 = _mm256_loadu_pd(&A[x - 1 - STRIDE * 3][y]);
								v10 = _mm256_loadu_pd(&A[x - 2 - STRIDE * 2][y + 1]);
								Input_high_3(v9, v2, in_4);							
								Compute_1vector(    v6, \
												v4, v7, v10, \
													v8);
								Input_Output_keep_src_2(out,v5,v7,in_0);
								_mm256_storeu_pd(&A[x - STRIDE * 2][y + 1], v5);




								v1 = _mm256_loadu_pd(&A[x - 2 - STRIDE * 3][y]);
								if(y + VECLEN <= myybeg - VECLEN + 1 + myyb - VECLEN){
									v5 = _mm256_loadu_pd(&A[x - 1][y + VECLEN + 3]);
									Input_high_4(v0, v5, in_4);
								} else {
									v0 = _mm256_set_pd( A[x - 3 - STRIDE * 3]	 [y - 1 + VECLEN], \
														A[x - 3 - STRIDE * 2]	 [y + 0 + VECLEN], \
														A[x - 3 - STRIDE]	 	 [y + 1 + VECLEN], \
														A[x - 3]			 	 [y + 2 + VECLEN]); 
								}
								Compute_1vector(    v9, \
												v7, v10, v1, \
													v11);
								Input_Output_keep_src_3(out,v8,v10,in_0);
								_mm256_storeu_pd(&A[x - STRIDE * 3][y], v8);



								if(y + VECLEN <= myybeg - VECLEN + 1 + myyb - VECLEN){
									in_4 = _mm256_loadu_pd(&A[x - 1 - STRIDE * VECLEN][y + VECLEN]);
									v8 = _mm256_loadu_pd(&A[x - 1 - STRIDE][y + VECLEN + 2]);
									v4 = _mm256_loadu_pd(&A[x - 2][y + VECLEN + 3]);
									Input_high_1(v3, v8, in_4);							
								} else {
									v4 = _mm256_set_pd( A[x - 2 - STRIDE * 3]	 [y + VECLEN], \
														A[x - 2 - STRIDE * 2]	 [y + VECLEN + 1], \
														A[x - 2 - STRIDE]	 	 [y + VECLEN + 2], \
														A[x - 2]			 	 [y + VECLEN + 3]); 
								}
								Compute_1vector(     v0, \
												v10, v1, v4, \
													v2);

								Input_Output_keep_src_4(out,v0,v1,in_0);							
								
								_mm256_storeu_pd(&A[x - STRIDE * 4][y], out);
							}

							_mm256_storeu_pd(tmp, v1);
							A[x - STRIDE]    [y + 2] = tmp[0];
							A[x - STRIDE * 2][y + 1] = tmp[1];
							A[x - STRIDE * 3][y + 0] = tmp[2];
							A[x - STRIDE * 4][y - 1] = tmp[3];	

							for(y += 1; y <= myybeg - VECLEN + 1 + myyb; y++){

								v3 = _mm256_set_pd( A[x - 3 - STRIDE * 3]	 [y - 1], \
													A[x - 3 - STRIDE * 2]	 [y - 1 + 1], \
													A[x - 3 - STRIDE]	 	 [y - 1 + 2], \
													A[x - 3]			 	 [y - 1 + 3]); 
								v7 = _mm256_set_pd( A[x - 2 - STRIDE * 3]	 [y], \
													A[x - 2 - STRIDE * 2]	 [y + 1], \
													A[x - 2 - STRIDE]	 	 [y + 2], \
													A[x - 2]			 	 [y + 3]); 
								v5 = _mm256_set_pd( A[x - 1 - STRIDE * 3]	 [y - 1], \
													A[x - 1 - STRIDE * 2]	 [y - 1 + 1], \
													A[x - 1 - STRIDE]	 	 [y - 1 + 2], \
													A[x - 1]			 	 [y - 1 + 3]);

								Compute_1vector(    v3, \
												v1, v4, v7, \
													v5);
								_mm256_storeu_pd(tmp, v4);

								A[x - STRIDE]    [y + 2] = tmp[0];
								A[x - STRIDE * 2][y + 1] = tmp[1];
								A[x - STRIDE * 3][y + 0] = tmp[2];
								A[x - STRIDE * 4][y - 1] = tmp[3];

								v1 = v4;
								v4 = v7;
							}
						} 
						//		if myxb != xb, process the bottom part of the block that is not updated by vector codes.
						for(x = myxbeg + myxb - STRIDE; x < myxbeg + myxb; x++){
							for ( y = myybeg - VECLEN + 1; y <= myybeg - VECLEN + 1 + myyb - VECLEN; y += VECLEN ){
								v1 = _mm256_loadu_pd(&A[x]					[y + 3]);
								v2 = _mm256_loadu_pd(&A[x  - STRIDE]		[y + 2]);
								v3 = _mm256_loadu_pd(&A[x  - STRIDE * 2]	[y + 1]);
								v4 = _mm256_loadu_pd(&A[x  - STRIDE * 3]	[y]);
								transpose(v1, v2, v3, v4, v5, v6);
								_mm256_storeu_pd(&A[x]					[y + 3], v1);
								_mm256_storeu_pd(&A[x  - STRIDE]		[y + 2], v2);
								_mm256_storeu_pd(&A[x  - STRIDE * 2]	[y + 1], v3);
								_mm256_storeu_pd(&A[x  - STRIDE * 3]	[y], v4);
							}
						}

						//		33221100|0
						//		44443211|0

						for( t = tt ; t < tt + VECLEN; t++){
							for(x = myxbeg + myxb - STRIDE * (t - tt + 1); x < min(NX + XSTART, xbeg + xb - (t - tt)); x++){
								for ( y = myybeg - (t - tt); y < myybeg - (t - tt) + myyb; y++){									
									Compute_scalar(A, x, y);
								}
							}
						//		if myyb != yb, process the right part of the block that is not updated by vector codes.
							for(x = max(XSTART, xbeg - (t - tt)); x < min(NX + XSTART, xbeg + xb - (t - tt)); x++){
								for ( y = myybeg + myyb - (t - tt); y < min(NY + YSTART, ybeg + yb - (t - tt)); y++){									
									Compute_scalar(A, x, y);
								}
							}
						}
					}
				}		
				// note that it should be x = max(XSTART, xbeg - (t - (wave - xx - yy) * tb))
				// and y = max(YSTART, ybeg - (t - (wave - xx - yy) * tb))
				// but if (wave - xx - yy) * tb < 0 and tt = max(0, (wave - xx - yy) * tb) = 0
				// then this block contains no computation as (wave - xx - yy + 1) * tb) <= 0
				for (t = tt; t < min(T, (wave - xx - yy + 1) * tb); t++, xbeg--, ybeg--){
					for (x = max(XSTART, xbeg); x < min(NX + XSTART, xbeg + xb); x++) {
						for (y = max(YSTART, ybeg); y < min(NY + YSTART, ybeg + yb); y++) {
							Compute_scalar(A, x, y);
						}
					}
				}							
			}
		}
	}
#ifdef scalarratio
	printf("%f\n", (double) cnt /(double)((double) NX * (double) NY * (double) T));
#endif
}

	/*        
	The following numbers 1, 2, 3, 4 show the region that is updated in the corresponding time steps.

	4444444444444444 
	44444444444444443 
	444444444444444432				
	4444444444444444321
	4444444444444444321
	4444444444444444321
	4444444444444444321
	 333333333333333321
	  22222222222222221
	   1111111111111111

	The following numbers 1, 2, 3, 4 show the values in time dimension after one block updating

	1111111111111111
	12222222222222221
	123333333333333321					
	1234444444444444321
	1234444444444444321
	1234444444444444321
	1234444444444444321
	1234444444444444321
	1234444444444444321
	 123333333333333321
	  12222222222222221
	   1111111111111111


					1111111111111111
					12222222222222221
					123333333333333321					
					1234444444444444321
					1234444444444444321
					1234444444444444321
					1234444444444444321
	1111111111111111 123333333333333321
	12222222222222221 12222222222222221
	123333333333333321 1111111111111111
	1234444444444444321
	1234444444444444321
	1234444444444444321
	1234444444444444321
	 123333333333333321
	  12222222222222221
	   1111111111111111


	12344444444444444444444444444444321
	12344444444444443333333333333333321
	12344444444444443222222222222222221
	12344444444444443211111111111111111fffffffffffffffff					
	12344444444444443210000000000000000
	12344444444444443210000000000000000
	12344444444444443210000000000000000
	12344444444444443210000000000000000
	12344444444444443210000000000000000
	12344444444444443210000000000000000
	12344444444444443210000000000000000
	12344444444444443210000000000000000
	12344444444444443210000000000000000
	12344444444444443210000000000000000
	 1233333333333333210000000000000000
	  122222222222222210000000000000000
	   11111111111111110000000000000000
        0000000000000000

	

	12344444444444444444444444444444321
	1234444444444444*333333333333333321
	12344444444444443333333333333333321
	12344444444444443*22222222222222221ffffffffffffffffffff					
	12344444444444443222222222222222221
	123444444444444432*1111111111111111
	12344444444444443211111111111111111
	1234444444444444321*000000000000000
	12344444444444443210000000000000000
	12344444444444443210#00000000000000
	12344444444444443210000000000000000
	12344444444444443210000000000000000
	12344444444444443210000000000000000
	12344444444444443210000000000000000
	 1233333333333333210000000000000000
	  122222222222222210000000000000000
	   11111111111111110000000000000000







1111111111111111
12222222222222221
123333333333333321					
1234444444444444321
1234444444444444321
1234444444444444321
1234444444444444321
1234444444444444321
1234444444444444321
 123333333333333321
  12222222222222221
   1111111111111111


444444444444444
3333333333333333
32222222222222222
-------------------
321111111111111111|
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0

444444444444444
3333333333333333
32222222222222222
-------------------
333333333333333321|
322222222222222221|0
321111111111111111|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0



444444444444444
3333333333333333
-------------------
32222222222222222 |
321111111111111111|
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
                    * ---> ybeg + yb
                   * -----> NY + YSTART = myybeg + myyb
444444444444444
3333333333333333
-------------------
33333333333333332 |
322222222222222221|
321111111111111111|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
321000000000000000|0
                    * ---> ybeg + yb
                   * -----> NY + YSTART = myybeg + myyb

4444444444444444
-------------------
44444444444444443 |
444444444444444432|
444444444444444432|1
444444444444444432|1
444444444444444432|1
444444444444444432|1
 33333333333333332|1
  2222222222222222|1
   111111111111111|1

1111111111111111
-------------------
12222222222222221 |
123333333333333321|					
123444444444444432|1
123444444444444432|1
123444444444444432|1
123444444444444432|1
123444444444444432|1
123444444444444432|1
 12333333333333332|1
  1222222222222222|1
   111111111111111|1

-------------------
111111111111111
1222222222222221 
12333333333333321					
123444444444444321
123444444444444321
123444444444444321
123444444444444321
123444444444444321
 12333333333333321
  1222222222222221
   111111111111111

*/

