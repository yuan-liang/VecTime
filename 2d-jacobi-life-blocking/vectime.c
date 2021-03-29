#include "define.h"

void vectime(int *A, int NX, int NY, int T, int xb, int yb, int tb)
{

	int(*B)[NX + 2 * XSTART][NY + 2 * YSTART] = (int(*)[NX + 2 * XSTART][NY + 2 * YSTART]) A;

	int tt, t, tv, x, xx, y, yy;

	tb -= tb % VECLEN_INT;
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

	int (* AV) [yb][VECLEN_INT] = (int(*)[yb][VECLEN_INT])alloc_extra_array(sizeof(int) * yb * VECLEN_INT * 3 * myid);

	for (wave = 0; wave < myceil(T, tb) + 1 + myceil(NY + T - 1, yb); wave++) { 

		#pragma omp parallel for private(myid, mylevel, tt, tv, t, x, y, xmin, xmax, ybeg, myybeg, myyb)  collapse(2) schedule(dynamic, 1)

		for (xx = 0; xx < max(nb0[0], nb0[1]); xx++) {

			for (yy = 0; yy < 2 * myceil(NY + T - 1, yb); yy++) {


				__m256i v_center_0, v_center_1, v_center_2; 
				__m256i v_x_plus_0, v_x_plus_1, v_x_plus_2; 
				__m256i v_x_minus_0, v_x_minus_1, v_x_minus_2;
				__m256i in, out, vzero, vone, vtwo, vthree, vrotatei_high2low;

				int zero[VECLEN_INT] =  {0, 0, 0, 0, 0, 0, 0, 0};
				int one[VECLEN_INT] =   {1, 1, 1, 1, 1, 1, 1, 1};
				int two[VECLEN_INT] =   {2, 2, 2, 2, 2, 2, 2, 2};
				int three[VECLEN_INT] = {3, 3, 3, 3, 3, 3, 3, 3};
				int rotatei_high2low[VECLEN_INT] = {7, 0, 1, 2, 3, 4, 5, 6};

				int tmp[8];

				myid = omp_get_thread_num();
				int (* BV0) [VECLEN_INT] = (int(*)[VECLEN_INT]) (AV + myid * 3 + 0);
				int (* BV1) [VECLEN_INT] = (int(*)[VECLEN_INT]) (AV + myid * 3 + 1);
				int (* BV2) [VECLEN_INT] = (int(*)[VECLEN_INT]) (AV + myid * 3 + 2);
				int (* Btmp [3]) [VECLEN_INT]  = {BV0, BV1, BV2};

				mylevel = (wave % 2 + yy) % 2;
                
				if (xx < nb0[mylevel]) {

					tt = -tb + (wave - yy) * tb;

                    ybeg = YSTART - wave * tb + yy * (yb + tb);
					
					for (tv = max(tt, 0); tv <= min(tt + 2 * tb, T) - VECLEN_INT; tv += VECLEN_INT, ybeg -= VECLEN_INT) {
						
                        if(ybeg - VECLEN_INT + 1 < YSTART){
					        myybeg 	= 	YSTART + VECLEN_INT - 1;
					        myyb	=	yb - (myybeg - ybeg);
					    } else {
					    	myybeg	=	ybeg;
					    	myyb	=	yb;
					    }
					    if (ybeg + yb >= YSTART + NY){	
					    	myyb -= ybeg + yb - (YSTART + NY);
					    }
                        if (myyb <= 2 * VECLEN_INT ) {
						    for (t = tv; t < tv + VECLEN_INT; t++){

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
						    //head
							for (t = tv; t < tv + VECLEN_INT - 1; t++){

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

						    for (t = tv; t < tv + VECLEN_INT; t++) {

						    	xmin = (mylevel == 1 && xx == 0) ? XSTART : (xright[mylevel] - Bx + xx * ix + myabs((tt + tb), (t + 1)) * XSLOPE);
						    	xmax = xmin + STRIDE * (VECLEN_INT - 1 - (t - tv));
						    	if (!(mylevel == 1 && xx == 0)) {
						    		xmax += ((tt + tb < tv + 1) ? 1 : -1) * (VECLEN_INT - 1 - (t - tv));
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

						    	for (y = myybeg - VECLEN_INT + 1; y <= myybeg - VECLEN_INT + 1 + myyb - VECLEN_INT; y += VECLEN_INT) {

									vloadi2(v_center_0, B[t%2    ][x + STRIDE * 7][y+7]);
									vloadi2(v_center_1, B[(t+1)%2][x + STRIDE * 6][y+6]);
									vloadi2(v_center_2, B[t%2    ][x + STRIDE * 5][y+5]);
									vloadi2(vrotatei_high2low, B[(t+1)%2][x + STRIDE * 4][y+4]);
									vloadi2(v_x_plus_0, B[t%2    ][x + STRIDE * 3][y+3]);
									vloadi2(v_x_plus_1, B[(t+1)%2][x + STRIDE * 2][y+2]);
									vloadi2(v_x_plus_2, B[t%2    ][x + STRIDE * 1][y+1]);
									vloadi2(vtwo, B[(t+1)%2][x             ][y]);
									transposei(v_center_0, v_center_1, v_center_2, vrotatei_high2low, v_x_plus_0, v_x_plus_1, v_x_plus_2, vtwo, in, out, vzero, vone, v_x_minus_0, v_x_minus_1, v_x_minus_2, vthree);
									vstorei( Btmp[x - xmin][y - (myybeg - VECLEN_INT + 1) + 7][0], vthree);
									vstorei( Btmp[x - xmin][y - (myybeg - VECLEN_INT + 1) + 6][0], v_x_minus_2);
									vstorei( Btmp[x - xmin][y - (myybeg - VECLEN_INT + 1) + 5][0], v_x_minus_1);
									vstorei( Btmp[x - xmin][y - (myybeg - VECLEN_INT + 1) + 4][0], v_x_minus_0);
									vstorei( Btmp[x - xmin][y - (myybeg - VECLEN_INT + 1) + 3][0], vone);
									vstorei( Btmp[x - xmin][y - (myybeg - VECLEN_INT + 1) + 2][0], vzero);
									vstorei( Btmp[x - xmin][y - (myybeg - VECLEN_INT + 1) + 1][0], out);
									vstorei( Btmp[x - xmin][y - (myybeg - VECLEN_INT + 1) + 0][0], in);
						    	}
						    }
						    for (x = xmin + XSLOPE; x < xmax - STRIDE * VECLEN_INT; x++) {

						    	y = myybeg - VECLEN_INT + 1;

								vzero = vloadi(zero[0]); 
								vone = vloadi(one[0]); 
								vtwo = vloadi(two[0]); 
								vthree = vloadi(three[0]);
								vrotatei_high2low = vloadi(rotatei_high2low[0]);

								vloadseti_blk(v_x_minus_0, B, t, x-XSLOPE, y-YSLOPE);  
								vloadseti_blk(v_center_0, B, t, x, y-YSLOPE);  
								vloadseti_blk(v_x_plus_0, B, t, x+XSLOPE, y-YSLOPE); 

								vloadi2(v_x_minus_1, BV0[0][0]);
								vloadi2(v_center_1,  BV1[0][0]);
								vloadi2(v_x_plus_1,  BV2[0][0]);

						    	vloadseti_blk(v_center_2, B, t, x + STRIDE, y);  
								
						    	for (; y <= myybeg - VECLEN_INT + 1 + myyb - VECLEN_INT; y += VECLEN_INT) {

						    		vloadi2(in, B[(t) % 2][x + STRIDE * VECLEN_INT][y + VECLEN_INT]); // the next x iter in vector

									vstorei(BV0[y - (myybeg - VECLEN_INT + 1)][0], v_center_2); 


									vloadi2(v_x_minus_2, BV0[y - (myybeg - VECLEN_INT + 1) + 1][0]);
									vloadi2(v_center_2,  BV1[y - (myybeg - VECLEN_INT + 1) + 1][0]); 
									vloadi2(v_x_plus_2,  BV2[y - (myybeg - VECLEN_INT + 1) + 1][0]); 
									Compute_1vector(v_x_minus_2, v_center_2, v_x_plus_2,\
													v_x_minus_1, v_center_1, v_x_plus_1,\
													v_x_minus_0, v_center_0, v_x_plus_0);
									Input_Output_i_1(out, v_center_0, in);
									vstorei(BV0[y - (myybeg - VECLEN_INT + 1) + 1][0], v_center_0); 


									vloadi2(v_x_minus_0, BV0[y - (myybeg - VECLEN_INT + 1) + 2][0]);
									vloadi2(v_center_0,  BV1[y - (myybeg - VECLEN_INT + 1) + 2][0]);
									vloadi2(v_x_plus_0,  BV2[y - (myybeg - VECLEN_INT + 1) + 2][0]);
									Compute_1vector(v_x_minus_0, v_center_0, v_x_plus_0,\
													v_x_minus_2, v_center_2, v_x_plus_2,\
													v_x_minus_1, v_center_1, v_x_plus_1); 
									Input_Output_i_2(out, v_center_1, in);
									vstorei(BV0[y - (myybeg - VECLEN_INT + 1) + 2][0], v_center_1); 
												


									vloadi2(v_x_minus_1, BV0[y - (myybeg - VECLEN_INT + 1) + 3][0]);
									vloadi2(v_center_1,  BV1[y - (myybeg - VECLEN_INT + 1) + 3][0]);
									vloadi2(v_x_plus_1,  BV2[y - (myybeg - VECLEN_INT + 1) + 3][0]);
									Compute_1vector(v_x_minus_1, v_center_1, v_x_plus_1,\
													v_x_minus_0, v_center_0, v_x_plus_0,\
													v_x_minus_2, v_center_2, v_x_plus_2); 
									Input_Output_i_3(out, v_center_2, in);	
									vstorei(BV0[y - (myybeg - VECLEN_INT + 1) + 3][0], v_center_2); 
																			

									vloadi2(v_x_minus_2, BV0[y - (myybeg - VECLEN_INT + 1) + 4][0]);
									vloadi2(v_center_2,  BV1[y - (myybeg - VECLEN_INT + 1) + 4][0]);
									vloadi2(v_x_plus_2,  BV2[y - (myybeg - VECLEN_INT + 1) + 4][0]);
									Compute_1vector(v_x_minus_2, v_center_2, v_x_plus_2,\
													v_x_minus_1, v_center_1, v_x_plus_1,\
													v_x_minus_0, v_center_0, v_x_plus_0);
									Input_Output_i_4(out, v_center_0, in);
               						vstorei(BV0[y - (myybeg - VECLEN_INT + 1) + 4][0], v_center_0); 


									vloadi2(v_x_minus_0, BV0[y - (myybeg - VECLEN_INT + 1) + 5][0]);
									vloadi2(v_center_0,  BV1[y - (myybeg - VECLEN_INT + 1) + 5][0]);
									vloadi2(v_x_plus_0,  BV2[y - (myybeg - VECLEN_INT + 1) + 5][0]);
									Compute_1vector(v_x_minus_0, v_center_0, v_x_plus_0,\
													v_x_minus_2, v_center_2, v_x_plus_2,\
													v_x_minus_1, v_center_1, v_x_plus_1); 
									Input_Output_i_5(out, v_center_1, in);
                					vstorei(BV0[y - (myybeg - VECLEN_INT + 1) + 5][0], v_center_1); 


									vloadi2(v_x_minus_1, BV0[y - (myybeg - VECLEN_INT + 1) + 6][0]);
									vloadi2(v_center_1,  BV1[y - (myybeg - VECLEN_INT + 1) + 6][0]);
									vloadi2(v_x_plus_1,  BV2[y - (myybeg - VECLEN_INT + 1) + 6][0]);
									Compute_1vector(v_x_minus_1, v_center_1, v_x_plus_1,\
													v_x_minus_0, v_center_0, v_x_plus_0,\
													v_x_minus_2, v_center_2, v_x_plus_2); 
									Input_Output_i_6(out, v_center_2, in);	
               						 vstorei(BV0[y - (myybeg - VECLEN_INT + 1) + 6][0], v_center_2);


									vloadi2(v_x_minus_2, BV0[y - (myybeg - VECLEN_INT + 1) + 7][0]);
									vloadi2(v_center_2,  BV1[y - (myybeg - VECLEN_INT + 1) + 7][0]);
									vloadi2(v_x_plus_2,  BV2[y - (myybeg - VECLEN_INT + 1) + 7][0]);
									Compute_1vector(v_x_minus_2, v_center_2, v_x_plus_2,\
													v_x_minus_1, v_center_1, v_x_plus_1,\
													v_x_minus_0, v_center_0, v_x_plus_0);
									Input_Output_i_7(out, v_center_0, in);
                					vstorei(BV0[y - (myybeg - VECLEN_INT + 1) + 7][0], v_center_0); 





									if(y + VECLEN_INT <= myybeg - VECLEN_INT + 1 + myyb - VECLEN_INT){
										vloadi2(v_x_minus_0, BV0[y - (myybeg - VECLEN_INT + 1) + 8][0]);
										vloadi2(v_center_0,  BV1[y - (myybeg - VECLEN_INT + 1) + 8][0]);
										vloadi2(v_x_plus_0,  BV2[y - (myybeg - VECLEN_INT + 1) + 8][0]);
									} else {
										vloadseti_blk(v_x_minus_0, B, t, x-XSLOPE, y+VECLEN_INT);  
										vloadseti_blk(v_center_0, B, t, x, y+VECLEN_INT);  
										vloadseti_blk(v_x_plus_0, B, t, x+XSLOPE, y+VECLEN_INT); 
									}
									Compute_1vector(v_x_minus_0, v_center_0, v_x_plus_0,\
													v_x_minus_2, v_center_2, v_x_plus_2,\
													v_x_minus_1, v_center_1, v_x_plus_1); 


									if (y >  myybeg - VECLEN_INT + 1 + myyb - VECLEN_INT - VECLEN_INT) {
										vstorei(tmp, v_center_1);
										B[(t) 	% 2][x + STRIDE * 0][y + VECLEN_INT - 1] = tmp[7];
										B[(t+1) % 2][x + STRIDE * 1][y + VECLEN_INT    ] = tmp[6];
										B[(t) 	% 2][x + STRIDE * 2][y + VECLEN_INT + 1] = tmp[5];
										B[(t+1) % 2][x + STRIDE * 3][y + VECLEN_INT + 2] = tmp[4];
										B[(t) 	% 2][x + STRIDE * 4][y + VECLEN_INT + 3] = tmp[3];
										B[(t+1) % 2][x + STRIDE * 5][y + VECLEN_INT + 4] = tmp[2];
										B[(t) 	% 2][x + STRIDE * 6][y + VECLEN_INT + 5] = tmp[1];
										B[(t+1) % 2][x + STRIDE * 7][y + VECLEN_INT + 6] = tmp[0];
									}
									Input_Output_i_8(out, v_center_1, in);
									vstorei(B[(t)%2][x][y], out);


									in = v_center_1;
									v_x_minus_1 = v_x_minus_0;
									v_center_1 = v_center_0;
									v_x_plus_1 = v_x_plus_0;
									v_x_minus_0 = v_x_minus_2;
									v_center_0 = v_center_2;
									v_x_plus_0 = v_x_plus_2;
									v_center_2 = in;
						    	}
								if((y >= myybeg - VECLEN_INT + 1 + myyb)){
									vstorei(tmp, v_x_minus_2);
									B[(t+1) % 2][x - 1 + STRIDE * 0][y - 1] = tmp[7];
									B[(t) 	% 2][x - 1 + STRIDE * 1][y    ] = tmp[6];
									B[(t+1) % 2][x - 1 + STRIDE * 2][y + 1] = tmp[5];
									B[(t) 	% 2][x - 1 + STRIDE * 3][y + 2] = tmp[4];
									B[(t+1) % 2][x - 1 + STRIDE * 4][y + 3] = tmp[3];
									B[(t) 	% 2][x - 1 + STRIDE * 5][y + 4] = tmp[2];
									B[(t+1) % 2][x - 1 + STRIDE * 6][y + 5] = tmp[1];
									B[(t) 	% 2][x - 1 + STRIDE * 7][y + 6] = tmp[0];
								}
						    	for ( ; y < myybeg - VECLEN_INT + 1 + myyb; y++) {
									vloadseti_blk(v_x_minus_2, B, t, x-XSLOPE, y+1);  
									vloadseti_blk(v_center_2, B, t, x, y+1);  
									vloadseti_blk(v_x_plus_2, B, t, x+XSLOPE, y+1); 

									Compute_1vector(v_x_minus_2, v_center_2, v_x_plus_2,\
													v_x_minus_1, v_center_1, v_x_plus_1,\
													v_x_minus_0, v_center_0, v_x_plus_0); 
									vstorei(tmp[0], v_center_0);

									B[(t+1)%2][x + STRIDE * 7][y + 7] = tmp[0];
									B[t%2    ][x + STRIDE * 6][y + 6] = tmp[1];
									B[(t+1)%2][x + STRIDE * 5][y + 5] = tmp[2];
									B[t%2    ][x + STRIDE * 4][y + 4] = tmp[3];
									B[(t+1)%2][x + STRIDE * 3][y + 3] = tmp[4];
									B[t%2    ][x + STRIDE * 2][y + 2] = tmp[5];
									B[(t+1)%2][x + STRIDE * 1][y + 1] = tmp[6];
									B[t%2    ][x             ][y - 0] = tmp[7];

									v_x_minus_0 = v_x_minus_1;
									v_center_0 = v_center_1;
									v_x_plus_0 = v_x_plus_1;
									v_x_minus_1 = v_x_minus_2;
									v_center_1 = v_center_2;
									v_x_plus_1 = v_x_plus_2;
						    	}

								Btmp[0] = BV0;
								BV0 = BV1;
								BV1 = BV2;
								BV2 = Btmp[0]; 
							}
							Btmp [0] = BV0;
							Btmp [1] = BV1;
							Btmp [2] = BV2;

						    for (; x < xmax - STRIDE * VECLEN_INT + 3; x++) {

						    	for (y = myybeg - VECLEN_INT + 1; y <= myybeg - VECLEN_INT + 1 + myyb - VECLEN_INT; y += VECLEN_INT) {
									vloadi2(vtwo, Btmp[x - (xmax - STRIDE * VECLEN_INT)][(y - (myybeg - VECLEN_INT + 1)) + 7][0]);
									vloadi2(v_x_plus_2, Btmp[x - (xmax - STRIDE * VECLEN_INT)][(y - (myybeg - VECLEN_INT + 1)) + 6][0]);
									vloadi2(v_x_plus_1, Btmp[x - (xmax - STRIDE * VECLEN_INT)][(y - (myybeg - VECLEN_INT + 1)) + 5][0]);
									vloadi2(v_x_plus_0, Btmp[x - (xmax - STRIDE * VECLEN_INT)][(y - (myybeg - VECLEN_INT + 1)) + 4][0]);
									vloadi2(vrotatei_high2low, Btmp[x - (xmax - STRIDE * VECLEN_INT)][(y - (myybeg - VECLEN_INT + 1)) + 3][0]);
									vloadi2(v_center_2, Btmp[x - (xmax - STRIDE * VECLEN_INT)][(y - (myybeg - VECLEN_INT + 1)) + 2][0]);
									vloadi2(v_center_1, Btmp[x - (xmax - STRIDE * VECLEN_INT)][(y - (myybeg - VECLEN_INT + 1)) + 1][0]);
									vloadi2(v_center_0, Btmp[x - (xmax - STRIDE * VECLEN_INT)][(y - (myybeg - VECLEN_INT + 1)) + 0][0]);
									transposei(v_center_0, v_center_1, v_center_2, vrotatei_high2low, v_x_plus_0, v_x_plus_1, v_x_plus_2, vtwo, in, out, vzero, vone, v_x_minus_0, v_x_minus_1, v_x_minus_2, vthree);
									vstorei( B[t%2    ][x - XSLOPE + STRIDE *7][y+7], in);
									vstorei( B[(t+1)%2][x - XSLOPE + STRIDE *6][y+6], out);
									vstorei( B[t%2    ][x - XSLOPE + STRIDE *5][y+5], vzero);
									vstorei( B[(t+1)%2][x - XSLOPE + STRIDE *4][y+4], vone);
									vstorei( B[t%2    ][x - XSLOPE + STRIDE *3][y+3], v_x_minus_0);
									vstorei( B[(t+1)%2][x - XSLOPE + STRIDE *2][y+2], v_x_minus_1);
									vstorei( B[t%2    ][x - XSLOPE + STRIDE *1][y+1], v_x_minus_2);
									vstorei( B[(t+1)%2][x - XSLOPE            ][y], vthree);
						    	}
						    }

                            // tail
						    xmin = xmax - STRIDE * VECLEN_INT;
						    for (t = tv; t < tv + VECLEN_INT; t++) {

								xmax = (mylevel == 1 && xx == nb0[1] - 1) ? NX + XSTART : (xright[mylevel] + xx * ix - myabs((tt + tb), (t + 1)) * XSLOPE);
						    	
								for (x = xmin + STRIDE * (VECLEN_INT - 1 - (t - tv)); x < xmax; x++) {

						    		#pragma ivdep
						    		#pragma vector always
						    		for (y = myybeg - (t - tv); y < myybeg - (t - tv) + myyb; y++) {
						    			Compute_scalar(B, t, x, y);
						    		}
						    	}
						    }
							for (t = tv; t < tv + VECLEN_INT; t++){

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
