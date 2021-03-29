#include "define.h"

void vectime(double *A, int NX, int NY, int T, int xb, int yb, int tb)
{

	double(*B)[NX + 2 * XSTART][NY + 2 * YSTART] = (double(*)[NX + 2 * XSTART][NY + 2 * YSTART]) A;

	int tt, t, tv, x, xx, y, yy;

	tb -= tb % 4;
	int Bx = xb;
	int bx = Bx - 2 * tb * XSLOPE;
	int ix = Bx + bx;
	int nb0[2] = {myfloor(NX - Bx, ix), myfloor(NX - Bx, ix) + 1};
	int nrestpoints = NX % ix;
	int bx_first_B1 = (Bx + nrestpoints) / 2;
	int bx_last_B1 = (Bx + nrestpoints) - bx_first_B1;
	int xright[2] = {bx_first_B1 + Bx + XSTART, bx_first_B1 + (Bx - bx) / 2 + XSTART};

	int wave;
	int mylevel;
	int xmin, xmax, ybeg;
    int myybeg, myyb;

	int myid = omp_get_max_threads();

	double (* AV) [yb][VECLEN] = (double(*)[yb][VECLEN])alloc_extra_array(sizeof(double) * yb * VECLEN * 3 * myid);

	for (wave = 0; wave < myceil(T, tb) + 1 + myceil(NY + T - 1, yb); wave++) { 

		#pragma omp parallel for private(myid, mylevel, tt, tv, t, x, y, xmin, xmax, ybeg, myybeg, myyb)  collapse(2) schedule(dynamic, 1)

		for (xx = 0; xx < max(nb0[0], nb0[1]); xx++) {

			for (yy = 0; yy < 2 * myceil(NY + T - 1, yb); yy++) {

				__m256d v_x_plus_0, v_x_plus_1, v_x_plus_2, v_x_plus_3; 
				__m256d v_x_minus_0, v_x_minus_1, v_x_minus_2, v_x_minus_3;
				vec v_center_0, v_center_1, v_center_2, v_center_3;
				vec in, out;
				SET_COFF;
				double tmp[4];

				myid = omp_get_thread_num();

				double (* BV0) [VECLEN] = (double(*)[VECLEN]) (AV + myid * 3 + 0);
				double (* BV1) [VECLEN] = (double(*)[VECLEN]) (AV + myid * 3 + 1);
				double (* BV2) [VECLEN] = (double(*)[VECLEN]) (AV + myid * 3 + 2);
				double (* Btmp [3]) [VECLEN]  = {BV0, BV1, BV2};

				mylevel = (wave % 2 + yy) % 2;
                
				if (xx < nb0[mylevel]) {

					tt = -tb + (wave - yy) * tb;

                    ybeg = YSTART - wave * tb + yy * (yb + tb);
					
					for (tv = max(tt, 0); tv <= min(tt + 2 * tb, T) - VECLEN; tv += VECLEN, ybeg -= VECLEN) {
						
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
                        if (myyb <= 2 * VECLEN ) {
						    for (t = tv; t < tv + VECLEN; t++){
						    	xmin = (mylevel == 1 && xx == 0) ?	             XSTART : (xright[mylevel] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
						        xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel]      + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);
								
						    	for (x = xmin; x < xmax; x++) {
						    		#pragma ivdep
									#pragma vector always
									for (y = max(YSTART, ybeg - (t - tv)); y < min(NY + YSTART, ybeg - (t - tv) + yb); y++) {
						    			Compute_scalar(B, t, x, y);
										
						    		}
						    	}
						    }
                        }
                        else {

							for (t = tv; t < tv + VECLEN - 1; t++){

						    	xmin = (mylevel == 1 && xx == 0) ?	             XSTART : (xright[mylevel] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
						        xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel]      + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);
								
						    	for (x = xmin; x < xmax; x++) {

						    		#pragma ivdep
									#pragma vector always
									for (y = max(YSTART, ybeg - (t - tv)); y < myybeg - (t - tv); y++) {
						    			Compute_scalar(B, t, x, y);

						    		}
						    	}
						    }

						    for (t = tv; t < tv + VECLEN; t++) {

						    	xmin = (mylevel == 1 && xx == 0) ? XSTART : (xright[mylevel] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
						    	xmax = xmin + STRIDE * (VECLEN - 1 - (t - tv));
						    	if (!(mylevel == 1 && xx == 0)) {
						    		xmax += ((tt + tb < tv + 1) ? 1 : -1) * (VECLEN - 1 - (t - tv));
						    	}
						    	for (x = xmin; x < xmax + 2; x++) {

						    		#pragma vector always
						    		#pragma ivdep
						    		for (y = myybeg - (t - tv); y < myybeg - (t - tv) + myyb ; y++) {
						    			Compute_scalar(B, t, x, y);
						    		}
						    	}
						    }

						    t = tv;

						    xmin += 1;
						    xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel] + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);
						    
							for (x = xmin; x < xmin + STRIDE + 1; x++) {

						    	for (y = myybeg - VECLEN + 1; y <= myybeg - VECLEN + 1 + myyb - VECLEN; y += VECLEN) {

						    		vload(v_center_3, B[(t + 1) % 2][x][y]);
						    		vload(v_center_2, B[(t) % 2][x + STRIDE][y+1]);
						    		vload(v_center_1, B[(t + 1) % 2][x + STRIDE * 2][y+2]);
						    		vload(v_center_0, B[(t) % 2][x + STRIDE * 3][y+3]);

						    		transpose(v_center_0, v_center_1, v_center_2, v_center_3, in, out);

									_mm256_storeu_pd(&Btmp[x - xmin][y - (myybeg - VECLEN + 1) + 0][0], v_center_0);
									_mm256_storeu_pd(&Btmp[x - xmin][y - (myybeg - VECLEN + 1) + 1][0], v_center_1);
									_mm256_storeu_pd(&Btmp[x - xmin][y - (myybeg - VECLEN + 1) + 2][0], v_center_2);
									_mm256_storeu_pd(&Btmp[x - xmin][y - (myybeg - VECLEN + 1) + 3][0], v_center_3);
						    	}
						    }
						    for (x = xmin + XSLOPE; x < xmax - STRIDE * VECLEN; x++) {

						    	y = myybeg - VECLEN + 1;

						    	v_center_0 = _mm256_set_pd(B[(t + 1) % 2][x             ][y - 1],\
                                                           B[(t)     % 2][x + STRIDE    ][y    ],\
                                                           B[(t + 1) % 2][x + STRIDE * 2][y + 1], \
                                                           B[(t)     % 2][x + STRIDE * 3][y + 2]);

						    	v_center_3    = _mm256_set_pd(B[(t + 1) % 2][x + STRIDE    ][y    ],\
                                                           B[(t)     % 2][x + STRIDE * 2][y + 1],\
                                                           B[(t + 1) % 2][x + STRIDE * 3][y + 2], \
                                                           B[(t)     % 2][x + STRIDE * 4][y + 3]);

								vload(v_center_1, BV1[0][0]);

						    	for (; y <= myybeg - VECLEN + 1 + myyb - VECLEN; y += VECLEN) {

						    		vload(in, B[(t) % 2][x + STRIDE * 4][y + VECLEN]);


									vloada(v_x_minus_0, BV0[y - ( myybeg - VECLEN + 1 )][0]);
									vloada(v_x_plus_0,  BV2[y - ( myybeg - VECLEN + 1 )][0]);
									vloada(v_center_2,  BV1[y - ( myybeg - VECLEN + 1 ) + 1][0]);
									vstorea(BV0[y - ( myybeg - VECLEN + 1 )][0], v_center_3);
						    		Compute_1vector(v_center_0, v_center_1, v_center_2, v_x_minus_0, v_x_plus_0);
						    		Input_Output_1(out, v_center_0, in);

									 
									vloada(v_x_minus_1, BV0[y - ( myybeg - VECLEN + 1 ) + 1][0]);
									vloada(v_x_plus_1,  BV2[y - ( myybeg - VECLEN + 1 ) + 1][0]);
									vloada(v_center_3,  BV1[y - ( myybeg - VECLEN + 1 ) + 2][0]);
									vstorea(BV0[y - ( myybeg - VECLEN + 1 ) + 1][0], v_center_0);
						    		Compute_1vector(v_center_1, v_center_2, v_center_3, v_x_minus_1, v_x_plus_1);
						    		Input_Output_2(out, v_center_1, in);


									vloada(v_x_minus_2, BV0[y - ( myybeg - VECLEN + 1 ) + 2][0]);
									vloada(v_x_plus_2,  BV2[y - ( myybeg - VECLEN + 1 ) + 2][0]);
									vloada(v_center_0,  BV1[y - ( myybeg - VECLEN + 1 ) + 3][0]);
									vstore(BV0[y - ( myybeg - VECLEN + 1 ) + 2][0], v_center_1);
						    		Compute_1vector(v_center_2, v_center_3, v_center_0, v_x_minus_2, v_x_plus_2);
						    		Input_Output_3(out, v_center_2, in);


									vloada(v_x_minus_3, BV0[y - ( myybeg - VECLEN + 1 ) + 3][0]);
									vloada(v_x_plus_3,  BV2[y - ( myybeg - VECLEN + 1 ) + 3][0]);
									v_center_1 = ( y > myybeg - VECLEN + 1 + myyb - VECLEN - VECLEN ) ? \
																_mm256_set_pd(	B[(t + 1) % 2][x][y + VECLEN], \
																				B[(t) % 2][x + STRIDE][y + VECLEN + 1], \
																				B[(t + 1) % 2][x + STRIDE * 2][y + VECLEN + 2], \
																				B[(t) % 2][x + STRIDE * 3][y + VECLEN + 3])\
															: _mm256_load_pd(&BV1[y - ( myybeg - VECLEN + 1 ) + 4][0])  ;
									vstorea(BV0[y - ( myybeg - VECLEN + 1 ) + 3][0], v_center_2);
						    		Compute_1vector(v_center_3, v_center_0, v_center_1, v_x_minus_3, v_x_plus_3); //4th
									
									if (y >  myybeg - VECLEN + 1 + myyb - VECLEN -VECLEN) {
										_mm256_storeu_pd(tmp, v_center_3);
										B[(t) 	% 2][x + STRIDE * 0][y + VECLEN - 1] = tmp[3];
										B[(t+1) % 2][x + STRIDE * 1][y + VECLEN    ] = tmp[2];
										B[(t) 	% 2][x + STRIDE * 2][y + VECLEN + 1] = tmp[1];
										B[(t+1) % 2][x + STRIDE * 3][y + VECLEN + 2] = tmp[0];
									}
						    		Input_Output_4(out, v_center_3, in);
									
						    		vstore(B[(t) % 2][x][y], out);
						    	}
								if((y >= myybeg - VECLEN + 1 + myyb)){
									_mm256_storeu_pd(tmp, v_x_minus_3);
										
									B[(t ) % 2][x-1 + STRIDE * 3][y+2] = tmp[0];
									B[(t + 1) % 2][x-1 + STRIDE * 2][y+1] = tmp[1];
									B[(t) % 2][x-1 + STRIDE * 1][y] = tmp[2];
									B[(t + 1) % 2][x-1][y-1] = tmp[3];
								}
						    	for ( ; y < myybeg - VECLEN + 1 + myyb; y++) {

						    		v_x_minus_1 = _mm256_set_pd(B[(t + 1) % 2][x - 1][y], \
                                                                B[(t) % 2][x - 1 + STRIDE][y+1], \
                                                                B[(t + 1) % 2][x - 1 + STRIDE * 2][y+2], \
                                                                B[(t) % 2][x - 1 + STRIDE * 3][y+3]);

						    		v_x_plus_1  = _mm256_set_pd(B[(t + 1) % 2][x + 1][y], \
                                                                B[(t) % 2][x + 1 + STRIDE][y+1], \
                                                                B[(t + 1) % 2][x + 1 + STRIDE * 2][y+2], \
                                                                B[(t) % 2][x + 1 + STRIDE * 3][y+3]);

						    		v_center_2  = _mm256_set_pd(B[(t + 1) % 2][x][y + 1], \
                                                                B[(t) % 2][x + STRIDE][y + 1 +1], \
                                                                B[(t + 1) % 2][x + STRIDE * 2][y + 1 +2], \
                                                                B[(t) % 2][x + STRIDE * 3][y + 1 +3]);

						    		Compute_1vector(v_center_0, v_center_1, v_center_2, v_x_minus_1, v_x_plus_1);

						    		_mm256_storeu_pd(tmp, v_center_0);
									
						    		B[(t + 1) % 2][x + STRIDE * 3][y+3] = tmp[0];
						    		B[t % 2][x + STRIDE * 2][y+2] = tmp[1];
						    		B[(t + 1) % 2][x + STRIDE * 1][y+1] = tmp[2];
						    		B[t % 2][x][y] = tmp[3];

						    		v_center_0 = v_center_1;
						    		v_center_1 = v_center_2;
						    	}
								Btmp[0] = BV0;
								BV0 = BV1;
								BV1 = BV2;
								BV2 = Btmp[0];   
						    }
							Btmp [0] = BV0;
							Btmp [1] = BV1;
							Btmp [2] = BV2;
						    for (; x < xmax - STRIDE * VECLEN + 3; x++) {

						    	for (y = myybeg - VECLEN + 1; y <= myybeg - VECLEN + 1 + myyb - VECLEN; y += VECLEN) {

									v_center_0 = _mm256_loadu_pd(&Btmp[x - (xmax - STRIDE * VECLEN)][(y - (myybeg - VECLEN + 1))][0]);
									v_center_1 = _mm256_loadu_pd(&Btmp[x - (xmax - STRIDE * VECLEN)][(y - (myybeg - VECLEN + 1)) + 1][0]);
									v_center_2 = _mm256_loadu_pd(&Btmp[x - (xmax - STRIDE * VECLEN)][(y - (myybeg - VECLEN + 1)) + 2][0]);
									v_center_3 = _mm256_loadu_pd(&Btmp[x - (xmax - STRIDE * VECLEN)][(y - (myybeg - VECLEN + 1)) + 3][0]);  
				
						    		transpose(v_center_0, v_center_1, v_center_2, v_center_3, in, out);

						    		vstore(B[(t + 1) % 2][x - XSLOPE + STRIDE * 0][y], v_center_3);
						    		vstore(B[(t) % 2][x - XSLOPE + STRIDE * 1][y+1], v_center_2);
						    		vstore(B[(t + 1) % 2][x - XSLOPE + STRIDE * 2][y+2], v_center_1);
						    		vstore(B[(t) % 2][x - XSLOPE + STRIDE * 3][y+3], v_center_0);
						    	}
						    }

                            // tail
						    xmin = xmax - STRIDE * VECLEN;
						    for (t = tv; t < tv + VECLEN; t++) {

								xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel] + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);
						    	
								for (x = xmin + STRIDE * (VECLEN - 1 - (t - tv)); x < xmax; x++) {

						    		#pragma ivdep
						    		#pragma vector always
						    		for (y = myybeg - (t - tv); y < myybeg - (t - tv) + myyb; y++) {
						    			Compute_scalar(B, t, x, y);
						    		}
						    	}
						    }

							for (t = tv; t < tv + VECLEN; t++){

						    	xmin = (mylevel == 1 && xx == 0) ?	             XSTART : (xright[mylevel] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
						        xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel]      + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);
								
						    	for (x = xmin; x < xmax; x++) {

						    		#pragma ivdep
									#pragma vector always
									for (y = myybeg + myyb - (t - tv); y < min(NY + YSTART, ybeg + yb - (t - tv)); y++) {
						    			Compute_scalar(B, t, x, y);

						    		}
						    	}
						    }
                        }
					}

					//extra
					for (t = tv; t < min(tt + 2 * tb, T); t++) {
						
						xmin = (mylevel == 1 && xx == 0) ?	             XSTART : (xright[mylevel] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
						xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel]      + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);
						
						for (x = xmin; x < xmax; x++) {

							#pragma ivdep
							#pragma vector always
							for (y = max(YSTART, ybeg - (t - tv)); y < min(NY + YSTART, ybeg - (t - tv) + yb); y++) {
								Compute_scalar(B, t, x, y);
							}
						}
					}
				}
			}
		}
	}
	free_extra_array(AV);
}
