#include "define.h"

void vectime(double* A, int NX, int NY, int T) {
	double (* B)[NX + 2 * XSTART][ NY + 2 * YSTART] =  (double (*)[NX + 2 * XSTART][ NY + 2 * YSTART]) A;
    double tmp[4];

    int tt, t, x, xx, y;
    __m256d v_center_0, v_center_1, v_center_2, v_center_3; 
    __m256d v_x_plus_0, v_x_plus_1, v_x_plus_2, v_x_plus_3; 
    __m256d v_x_minus_0, v_x_minus_1, v_x_minus_2, v_x_minus_3;
    __m256d in, out;
	SET_COFF;


	double (* AV) [NY][VECLEN] = (double(*)[NY][VECLEN])alloc_extra_array(sizeof(double) * NY * VECLEN * 3);

	double (* BV0) [VECLEN] = (double (*) [VECLEN]) AV;
	double (* BV1) [VECLEN] = (double (*) [VECLEN]) (AV + 1);
    double (* BV2) [VECLEN] = (double (*) [VECLEN]) (AV + 2);

	double (* Btmp [3]) [VECLEN]  = {BV0, BV1, BV2};

	for ( tt = 0; tt <= T - VECLEN; tt += VECLEN){	

		for( t = tt ; t < tt + VECLEN - 1 ; t++){
			for ( x = XSTART; x < XSTART + STRIDE * (VECLEN - 1 - (t - tt)); x++) {
            #pragma ivdep
            #pragma vector always
                for ( y = YSTART; y < NY + YSTART; y++) {
                    Compute_scalar(B, t, x, y);
                }		
            }
		}

        t = tt;

        for(x = XSTART - XSLOPE; x <= XSTART +XSLOPE; x++){
			for ( y = YSTART; y <= NY + YSTART - VECLEN; y += VECLEN ){
				vload(v_center_0, B[t%2    ][x + STRIDE *3][y]);
				vload(v_center_1, B[(t+1)%2][x + STRIDE *2][y]);
				vload(v_center_2, B[t%2    ][x + STRIDE *1][y]);
				vload(v_center_3, B[(t+1)%2][x            ][y]);
				transpose(v_center_0, v_center_1, v_center_2, v_center_3, in, out);
				vstore( Btmp[x - (XSTART - XSLOPE)][(y - YSTART) + 3][0], v_center_3);
				vstore( Btmp[x - (XSTART - XSLOPE)][(y - YSTART) + 2][0], v_center_2);
				vstore( Btmp[x - (XSTART - XSLOPE)][(y - YSTART) + 1][0], v_center_1);
				vstore( Btmp[x - (XSTART - XSLOPE)][(y - YSTART)][0], v_center_0);
			}
		}        
                
        for ( x = XSTART ; x <= NX + XSTART - STRIDE * VECLEN ; x ++){
            y = YSTART;
            vloadset(v_x_minus_0, B[(t+1)%2][x-XSLOPE         ][y-YSLOPE], \
                          B[(t)%2  ][x-XSLOPE+STRIDE  ][y-YSLOPE], \
                          B[(t+1)%2][x-XSLOPE+STRIDE*2][y-YSLOPE], \
                          B[(t)%2  ][x-XSLOPE+STRIDE*3][y-YSLOPE]);

            vloadset(v_center_0, B[(t+1)%2][x         ][y-YSLOPE], \
                          B[(t)%2  ][x+STRIDE  ][y-YSLOPE], \
                          B[(t+1)%2][x+STRIDE*2][y-YSLOPE], \
                          B[(t)%2  ][x+STRIDE*3][y-YSLOPE]);

            vloadset(v_x_plus_0, B[(t+1)%2][x+XSLOPE         ][y-YSLOPE], \
                          B[(t)%2  ][x+XSLOPE+STRIDE  ][y-YSLOPE], \
                          B[(t+1)%2][x+XSLOPE+STRIDE*2][y-YSLOPE], \
                          B[(t)%2  ][x+XSLOPE+STRIDE*3][y-YSLOPE]);

            vload(v_x_minus_1, BV0[0][0]);
            vload(v_center_1, BV1[0][0]);
            vload(v_x_plus_1, BV2[0][0]);

            for ( y = YSTART; y <= NY + YSTART - VECLEN; y+= VECLEN) {

                vload(v_x_minus_3, BV0[(y - YSTART) + 2][0]);
                vload(v_x_minus_2, BV0[(y - YSTART) + 1][0]);         //(x, y) = (-1, -1) left

                vload(v_center_3, BV1[(y - YSTART) + 2][0]);
                vload(v_center_2, BV1[(y - YSTART) + 1][0]);                //(x, y) = (0, -1) mid 

                vload(v_x_plus_3, BV2[(y - YSTART) + 2][0]);
                vload(v_x_plus_2, BV2[(y - YSTART) + 1][0]);         //(x, y) = (1, -1) right

                vload(in, B[(t)%2][x+STRIDE*4][y]);   // the next x iter in vector

                Compute_1vector(v_x_minus_2, v_center_2, v_x_plus_2,\
                                v_x_minus_1, v_center_1, v_x_plus_1,\
                                v_x_minus_0, v_center_0, v_x_plus_0);   //1st   store the newest value to the left vec
                v_x_minus_0 = vrotate_high2low(v_x_minus_0);
                out =v_x_minus_0;
                v_x_minus_0 = _mm256_blend_pd(v_x_minus_0, in, 0b0001);
                vstore(BV0[(y - YSTART)][0], v_x_minus_0);   //for the compute of next x iteration


                Compute_1vector(v_x_minus_3, v_center_3, v_x_plus_3,\
                                v_x_minus_2, v_center_2, v_x_plus_2,\
                                v_x_minus_1, v_center_1, v_x_plus_1); 
                in = _mm256_permute_pd(in, 0b0101);
                v_x_minus_1 = vrotate_high2low(v_x_minus_1);
                shuffle(out, v_x_minus_1, 0b0000);
                v_x_minus_1 = _mm256_blend_pd(v_x_minus_1, in, 0b0001);
                vstore(BV0[(y - YSTART) + 1][0], v_x_minus_1);  //for the compute of next x iteration


                vload(v_x_minus_0, BV0[(y - YSTART) + 3][0]);
                vload(v_center_0, BV1[(y - YSTART) + 3][0]);
                vload(v_x_plus_0, BV2[(y - YSTART) + 3][0]);
                Compute_1vector(v_x_minus_0, v_center_0, v_x_plus_0,\
                                v_x_minus_3, v_center_3, v_x_plus_3,\
                                v_x_minus_2, v_center_2, v_x_plus_2); 
                out = _mm256_blend_pd(out, v_x_minus_2, 0b1000);
                v_x_minus_2 = _mm256_blend_pd(v_x_minus_2, in, 0b1000);
                v_x_minus_2 = vrotate_high2low(v_x_minus_2);	
                vstore(BV0[(y - YSTART) + 2][0], v_x_minus_2);

                out = _mm256_permute_pd(out, 0b0110);
                in = _mm256_permute_pd(in, 0b0101);


                if(y + VECLEN <= NY + YSTART - VECLEN){
                    vload(v_x_minus_1, BV0[(y - YSTART) + 4][0]);
                    vload(v_center_1, BV1[(y - YSTART) + 4][0]);
                    vload(v_x_plus_1, BV2[(y - YSTART) + 4][0]);

                } else {

                    vloadset(v_x_minus_1, B[(t+1)%2][x-XSLOPE         ][y+VECLEN], \
                                B[(t)%2  ][x-XSLOPE+STRIDE  ][y+VECLEN], \
                                B[(t+1)%2][x-XSLOPE+STRIDE*2][y+VECLEN], \
                                B[(t)%2  ][x-XSLOPE+STRIDE*3][y+VECLEN]);

                    vloadset(v_center_1, B[(t+1)%2][x         ][y+VECLEN], \
                                B[(t)%2  ][x+STRIDE  ][y+VECLEN], \
                                B[(t+1)%2][x+STRIDE*2][y+VECLEN], \
                                B[(t)%2  ][x+STRIDE*3][y+VECLEN]);

                    vloadset(v_x_plus_1, B[(t+1)%2][x+XSLOPE         ][y+VECLEN], \
                                B[(t)%2  ][x+XSLOPE+STRIDE  ][y+VECLEN], \
                                B[(t+1)%2][x+XSLOPE+STRIDE*2][y+VECLEN], \
                                B[(t)%2  ][x+XSLOPE+STRIDE*3][y+VECLEN]);
                }
                Compute_1vector(v_x_minus_1, v_center_1, v_x_plus_1,\
                                v_x_minus_0, v_center_0, v_x_plus_0,\
                                v_x_minus_3, v_center_3, v_x_plus_3); 
                out = _mm256_blend_pd(out, v_x_minus_3, 0b1000);
                v_x_minus_3 = _mm256_blend_pd(v_x_minus_3, in, 0b1000);
                v_x_minus_3 = vrotate_high2low(v_x_minus_3);
                vstore(BV0[(y - YSTART) + 3][0], v_x_minus_3);

                vstore(B[(t)%2][x][y], out);

            }
         
            for ( y += 1; y <= NY+YSTART; y++){
                vloadset(v_x_minus_2, B[(t+1)%2][x-XSLOPE         ][y], \
                              B[(t)%2  ][x-XSLOPE+STRIDE  ][y], \
                              B[(t+1)%2][x-XSLOPE+STRIDE*2][y], \
                              B[(t)%2  ][x-XSLOPE+STRIDE*3][y]);

                vloadset(v_center_2, B[(t+1)%2][x         ][y], \
                              B[(t)%2  ][x+STRIDE  ][y], \
                              B[(t+1)%2][x+STRIDE*2][y], \
                              B[(t)%2  ][x+STRIDE*3][y]);

                vloadset(v_x_plus_2, B[(t+1)%2][x+XSLOPE         ][y], \
                              B[(t)%2  ][x+XSLOPE+STRIDE  ][y], \
                              B[(t+1)%2][x+XSLOPE+STRIDE*2][y], \
                              B[(t)%2  ][x+XSLOPE+STRIDE*3][y]);

                Compute_1vector(v_x_minus_2, v_center_2, v_x_plus_2,\
                                v_x_minus_1, v_center_1, v_x_plus_1,\
                                v_x_minus_0, v_center_0, v_x_plus_0); 
                _mm256_storeu_pd(tmp, v_x_minus_0);

		    	B[(t+1)%2][x + STRIDE * 3][y - 1] = tmp[0];
		    	B[t%2    ][x + STRIDE * 2][y - 1] = tmp[1];
		    	B[(t+1)%2][x + STRIDE * 1][y - 1] = tmp[2];
		    	B[t%2    ][x             ][y - 1] = tmp[3];
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
        xx = x;
        for(x = xx - XSLOPE; x <= xx +XSLOPE; x++){
			for ( y = YSTART; y <= NY + YSTART - VECLEN; y += VECLEN ){
				vload(v_center_3, Btmp[x - (xx - XSLOPE)][(y - YSTART) + 3][0]);
				vload(v_center_2, Btmp[x - (xx - XSLOPE)][(y - YSTART) + 2][0]);
				vload(v_center_1, Btmp[x - (xx - XSLOPE)][(y - YSTART) + 1][0]);
				vload(v_center_0, Btmp[x - (xx - XSLOPE)][(y - YSTART)][0]);
				transpose(v_center_0, v_center_1, v_center_2, v_center_3, in, out);
				vstore( B[t%2    ][x + STRIDE *3][y], v_center_0);
				vstore( B[(t+1)%2][x + STRIDE *2][y], v_center_1);
				vstore( B[t%2    ][x + STRIDE *1][y], v_center_2);
				vstore( B[(t+1)%2][x            ][y], v_center_3);
			}
		} 
    
        for( t = tt ; t < tt + VECLEN ; t++){
			for ( x = xx + STRIDE * (VECLEN - 1 - (t - tt)); x < NX + XSTART; x++) {
                #pragma ivdep
                #pragma vector always
                for ( y = YSTART; y < NY + YSTART; y++) {
                    Compute_scalar(B, t, x, y);
                }		
            }
		}
        for ( y = 0; y < NY + YSTART * 2 ; y++) {
            B[1][XSTART - XSLOPE][y] = B[0][XSTART - XSLOPE][y];
        }
	}

	for (t = tt ; t < T; t++){
		for (x = XSTART; x < NX + XSTART; x++) {
            #pragma ivdep
            #pragma vector always
            for ( y = YSTART; y < NY + YSTART; y++) {
               Compute_scalar(B, t, x, y);
            }
		}
	}	

    free_extra_array(AV);
}
