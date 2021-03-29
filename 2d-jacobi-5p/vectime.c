#include "define.h"

void vectime(double* A, int NX, int NY, int T) {

    double (* B)[NX + 2 * XSTART][ NY + 2 * YSTART] =  (double (*)[NX + 2 * XSTART][ NY + 2 * YSTART]) A;
    double tmp[4];
    int tt, t = 0, x, xx, y, yy;
    __m256d v_x_plus_0, v_x_plus_1, v_x_plus_2, v_x_plus_3; 
    __m256d v_x_minus_0, v_x_minus_1, v_x_minus_2, v_x_minus_3;
    vec v_center_0, v_center_1, v_center_2, v_center_3; 
    vec in, out;
	SET_COFF;

	double (* AV) [NY+ 2 * YSTART][VECLEN] = (double(*)[NY+ 2 * YSTART][VECLEN])alloc_extra_array(sizeof(double) * (NY + 2 * YSTART) * VECLEN * 3);

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
    
        for(x = XSTART; x < XSTART + STRIDE + 1; x++){
            for ( y = YSTART; y <= NY + YSTART - VECLEN ; y+= VECLEN) {
                vload(v_center_3, B[(t+1)%2][x-XSLOPE         ][y]);
                vload(v_center_2, B[(t)%2  ][x-XSLOPE+STRIDE  ][y]);
                vload(v_center_1, B[(t+1)%2][x-XSLOPE+STRIDE*2][y]);
                vload(v_center_0, B[(t)%2  ][x-XSLOPE+STRIDE*3][y]);
                transpose(v_center_0, v_center_1, v_center_2, v_center_3, in, out);
				_mm256_storeu_pd(&Btmp[x - XSTART][(y - YSTART)][0], v_center_0);
				_mm256_storeu_pd(&Btmp[x - XSTART][(y - YSTART) + 1][0], v_center_1);
				_mm256_storeu_pd(&Btmp[x - XSTART][(y - YSTART) + 2][0], v_center_2);
				_mm256_storeu_pd(&Btmp[x - XSTART][(y - YSTART) + 3][0], v_center_3);
            }
        }

        for ( x = XSTART; x <= NX + XSTART - STRIDE * VECLEN ; x ++){

            y = YSTART;
            v_center_0 = _mm256_set_pd(B[(t+1)%2][x][y-1], B[(t)%2][x+STRIDE][y-1], B[(t+1)%2][x+STRIDE*2][y-1], B[(t)%2][x+STRIDE*3][y-1]);
            vload(v_center_1, BV1[0][0]);

            for (; y <= NY + YSTART - VECLEN ; y+= VECLEN) {

                vload(in, B[(t)%2][x+STRIDE*4][y]);   // the next x iter in vector

                vload(v_x_minus_0, BV0[(y - YSTART)][0]);
                vload(v_x_plus_0, BV2[(y - YSTART)][0]);
                vload(v_center_2, BV1[(y - YSTART) + 1][0]);
                Compute_1vector(v_center_0, v_center_1, v_center_2, v_x_minus_0, v_x_plus_0);   //1st   store the newest value to the left vec
                Input_Output_1(out,v_center_0,in);
                vstore(BV0[(y - YSTART)][0], v_center_0);

                vload(v_x_minus_1, BV0[(y - YSTART) + 1][0]);
                vload(v_x_plus_1, BV2[(y - YSTART) + 1][0]);
                vload(v_center_3, BV1[(y - YSTART) + 2][0]);
                Compute_1vector(v_center_1, v_center_2, v_center_3, v_x_minus_1, v_x_plus_1);   //2nd
                Input_Output_2(out,v_center_1,in);
                vstore(BV0[(y - YSTART) + 1][0], v_center_1);

                vload(v_x_minus_2, BV0[(y - YSTART) + 2][0]);
                vload(v_x_plus_2, BV2[(y - YSTART) + 2][0]);
                vload(v_center_0, BV1[(y - YSTART) + 3][0]);
                Compute_1vector(v_center_2, v_center_3, v_center_0, v_x_minus_2, v_x_plus_2);   //3rd
                Input_Output_3(out,v_center_2,in);
                vstore(BV0[(y - YSTART) + 2][0], v_center_2);

                vload(v_x_minus_3, BV0[(y - YSTART) + 3][0]);
                vload(v_x_plus_3, BV2[(y - YSTART) + 3][0]);
                v_center_1 = ( y > NY + YSTART - VECLEN - VECLEN ) ? (_mm256_set_pd(B[(t+1)%2][x][y+VECLEN], B[(t)%2][x+STRIDE][y+VECLEN], \
                                           B[(t+1)%2][x+STRIDE*2][y+VECLEN], B[(t)%2][x+STRIDE*3][y+VECLEN])) \
                                           : _mm256_loadu_pd(&BV1[(y - YSTART) + 4][0])  ;
                Compute_1vector(v_center_3, v_center_0, v_center_1, v_x_minus_3, v_x_plus_3);   //4th
                Input_Output_4(out,v_center_3,in);
                vstore(BV0[(y - YSTART) + 3][0], v_center_3);

                vstore(B[(t)%2][x][y], out);
            }
            for ( ; y < NY+YSTART; y++){
    
                v_x_minus_0 = _mm256_set_pd(B[(t+1)%2][x-1][y], B[(t)%2][x-1+STRIDE][y], B[(t+1)%2][x-1+STRIDE*2][y], B[(t)%2][x-1+STRIDE*3][y]);
                v_x_plus_0 = _mm256_set_pd(B[(t+1)%2][x+1][y], B[(t)%2][x+1+STRIDE][y], B[(t+1)%2][x+1+STRIDE*2][y], B[(t)%2][x+1+STRIDE*3][y]);
                v_center_2 = _mm256_set_pd(B[(t+1)%2][x][y+1], B[(t)%2][x+STRIDE][y+1], B[(t+1)%2][x+STRIDE*2][y+1], B[(t)%2][x+STRIDE*3][y+1]);

                Compute_1vector(v_center_0, v_center_1, v_center_2, v_x_minus_0, v_x_plus_0);

                _mm256_storeu_pd(tmp, v_center_0);
                B[(t+1)%2][x + STRIDE * 3][ y ] = tmp[0];
                B[t%2    ][x + STRIDE * 2][ y ] = tmp[1];
                B[(t+1)%2][x + STRIDE * 1][ y ] = tmp[2];
                B[t%2    ][x             ][ y ] = tmp[3];

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
        for(; x < NX + XSTART - STRIDE * VECLEN + 1 + 3; x++){
            for ( y = YSTART; y <= NY + YSTART - VECLEN ; y+= VECLEN) {
 				v_center_0 = _mm256_loadu_pd(&Btmp[x - (NX + XSTART - STRIDE * VECLEN + 1)][(y - YSTART)][0]);
				v_center_1 = _mm256_loadu_pd(&Btmp[x - (NX + XSTART - STRIDE * VECLEN + 1)][(y - YSTART) + 1][0]);
				v_center_2 = _mm256_loadu_pd(&Btmp[x - (NX + XSTART - STRIDE * VECLEN + 1)][(y - YSTART) + 2][0]);
				v_center_3 = _mm256_loadu_pd(&Btmp[x - (NX + XSTART - STRIDE * VECLEN + 1)][(y - YSTART) + 3][0]);               
                transpose(v_center_0, v_center_1, v_center_2, v_center_3, in, out);
                vstore(B[(t+1)%2][x-XSLOPE+STRIDE*0][y], v_center_3);
                vstore(B[(t)%2  ][x-XSLOPE+STRIDE*1][y], v_center_2);
                vstore(B[(t+1)%2][x-XSLOPE+STRIDE*2][y], v_center_1);
                vstore(B[(t)%2  ][x-XSLOPE+STRIDE*3][y], v_center_0);
            }
        }

        xx = NX + XSTART - STRIDE * VECLEN + 1;
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

	for ( ; t < T; t++){
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
