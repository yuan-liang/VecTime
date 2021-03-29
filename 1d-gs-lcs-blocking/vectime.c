#include "define.h"

int vectime(char * st, char * sx, int nt, int nx, int bt, int bx, int * lcs){

	int tbeg, xbeg, tt, xx, t, x, k;
    int mytbeg, myxbeg;
    int tsize, xsize;
	int lcs_t_minus_1_x_minus_1;
    int lcs_t_x_minus_1;
	int lcs_t_x;
	int nb;
	int wave;
	int nbnum;


    int (* lcsleft) [bt + 16] = (int (*) [bt + 16])malloc( sizeof(int) * myceil(nt, bt) * (bt + 16));
    memset((int *)lcsleft, 0, sizeof(int) * myceil(nt, bt) * (bt + 16));


    int nbx = myceil(nx, bx);
    int nbt = myceil(nt, bt);

	for (wave = 0; wave < nbx + nbt - 1; wave++) {

        xbeg = XSTART;
        if (wave >= nbt) {
            xbeg += (wave - nbt + 1) * bx;
        } 

        tbeg = XSTART + wave * bt;
        if (wave >= nbt - 1){
            tbeg = XSTART + (nbt - 1) * bt; 
        }
        
		nbnum = min((wave < nbt) ? (wave + 1) : nbt, myceil( nx + XSTART - xbeg, bx));
		
		#pragma omp parallel for schedule(dynamic, 1)  private(mytbeg, myxbeg, tsize, xsize, tt, xx, t, x, k, lcs_t_minus_1_x_minus_1, lcs_t_x_minus_1, lcs_t_x)

		for ( nb = 0; nb < nbnum; nb++){


		int head[STRIDE + 1][VECLEN_INT];
		__m256i v1, v2, v3, v4;
		__m256i vnewvalue, in, out;
		__m256i	vx1, vx2_1, vx2_2, vx2_3, in_x2;
		__m256i vrotatei_high2low, vone, mask;


            myxbeg = xbeg + nb * bx;
            mytbeg = tbeg - nb * bt;

            if (wave >= nbt - 1 && 0 == nb){
                tsize = nt + XSTART - tbeg;
            } else {
                tsize = bt;
            }
            xsize = min(nx + XSTART, myxbeg + bx) - myxbeg;


			t = mytbeg;

			for ( tt = mytbeg; tt <= mytbeg + tsize - VECLEN_INT; tt += VECLEN_INT)
			{
				if (xsize < (STRIDE + 1) * VECLEN_INT) break;

				for(k = myxbeg + STRIDE * (VECLEN_INT - 1) - 1; k <  myxbeg + STRIDE * VECLEN_INT; k++){
					head[k - (myxbeg + STRIDE * (VECLEN_INT - 1) - 1)][0] = lcs[k];
				}



				for (t = tt; t < tt + VECLEN_INT - 1; t++) {

						lcs_t_minus_1_x_minus_1 = lcsleft[mytbeg / bt][t - mytbeg];
						
						for ( x = myxbeg, k = myxbeg - XSTART + LCSSTART; x < myxbeg + STRIDE * (VECLEN_INT - (t - tt + 1)); x++, k++) {

							if (x == myxbeg) {
								lcs_t_x_minus_1 = lcsleft[mytbeg / bt][t - mytbeg + 1];
							} else {
								lcs_t_x_minus_1 = lcs[k - 1];
							}
							if ( st[t] == sx[x] ) {
								lcs_t_x = 1 + lcs_t_minus_1_x_minus_1;
							} else {
								lcs_t_x = max(lcs_t_x_minus_1, lcs[k]);
							} 

							lcs_t_minus_1_x_minus_1 = lcs[k];
							lcs[k] = lcs_t_x;
						}

						for(x = k - STRIDE - 1; x < k; x++){
							head[x - (k - STRIDE - 1)][t - tt + 1] = lcs[x];
						}
				}
				head[0][VECLEN_INT - 1] = lcsleft[mytbeg / bt] [t - mytbeg];
				
				v1 = vloadi(head[0][0]);
				v2 = vloadi(head[1][0]);
				v3 = vloadi(head[2][0]);
				v4 = vloadi(head[3][0]);


				head[STRIDE][0] = lcsleft[mytbeg / bt][t - mytbeg + 1];
				vnewvalue = vloadi(head[STRIDE][0]);
	
				int rotatei_low2high[VECLEN_INT] = {1, 2, 3, 4, 5, 6, 7, 0};
				vrotatei_high2low = vloadi(rotatei_low2high[0]);
				vnewvalue = vrotate_high2lowi(vnewvalue);

				int rotatei_high2low[VECLEN_INT] = {7, 0, 1, 2, 3, 4, 5, 6};
				vrotatei_high2low = vloadi(rotatei_high2low[0]);

				int one[VECLEN_INT] = {1, 1, 1, 1, 1, 1, 1, 1};
				vone = vloadi(one[0]);

				int headx2[STRIDE][VECLEN_INT];

				for ( t = 0; t < STRIDE; t++){

					for ( x = 0; x < VECLEN_INT; x++){

						headx2[t][VECLEN_INT - 1 - x] = sx[myxbeg + t + STRIDE * x];
					}			
				}

				vx2_1 = vloadi(headx2[0]);
				vx2_2 = vloadi(headx2[1]);
				vx2_3 = vloadi(headx2[2]);

				t = tt;
				vx1 = _mm256_setr_epi32(		(int) st[t + 0], \
												(int) st[t + 1], \
												(int) st[t + 2], \
												(int) st[t + 3], \
												(int) st[t + 4], \
												(int) st[t + 5], \
												(int) st[t + 6], \
												(int) st[t + 7]);

				for(x = myxbeg - XSTART + LCSSTART + STRIDE * VECLEN_INT; x <= myxbeg - XSTART + LCSSTART + xsize - VECLEN_INT; x += VECLEN_INT){

					in = vloadi(lcs[x]);

					in_x2 =  _mm256_setr_epi32(	(int) sx[x + 0 - LCSSTART + XSTART], \
												(int) sx[x + 1 - LCSSTART + XSTART], \
												(int) sx[x + 2 - LCSSTART + XSTART], \
												(int) sx[x + 3 - LCSSTART + XSTART], \
												(int) sx[x + 4 - LCSSTART + XSTART], \
												(int) sx[x + 5 - LCSSTART + XSTART], \
												(int) sx[x + 6 - LCSSTART + XSTART], \
												(int) sx[x + 7 - LCSSTART + XSTART]);

					mask = _mm256_cmpeq_epi32(vx1, vx2_1);

					vnewvalue = _mm256_max_epi32(vnewvalue, v2);
					vnewvalue = _mm256_blendv_epi8(vnewvalue, _mm256_add_epi32(v1, vone), mask);

					Input_Output_i_keep_src_1(out, v1, vnewvalue, in);
					Input_i_1(vx2_1,in_x2);


					mask = _mm256_cmpeq_epi32(vx1, vx2_2);
					vnewvalue = _mm256_max_epi32(vnewvalue, v3);
					vnewvalue = _mm256_blendv_epi8(vnewvalue, _mm256_add_epi32(v2, vone), mask);

					Input_Output_i_keep_src_2(out, v2, vnewvalue, in);
					Input_i_2(vx2_2,in_x2);

					mask = _mm256_cmpeq_epi32(vx1, vx2_3);
					vnewvalue = _mm256_max_epi32(vnewvalue, v4);
					vnewvalue = _mm256_blendv_epi8(vnewvalue, _mm256_add_epi32(v3, vone), mask);
					Input_Output_i_keep_src_3(out, v3, vnewvalue, in);
					Input_i_3(vx2_3,in_x2);

					mask = _mm256_cmpeq_epi32(vx1, vx2_1);
					vnewvalue = _mm256_max_epi32(vnewvalue, v1);
					vnewvalue = _mm256_blendv_epi8(vnewvalue, _mm256_add_epi32(v4, vone), mask);
					Input_Output_i_keep_src_4(out, v4, vnewvalue, in);
					Input_i_4(vx2_1,in_x2);


					mask = _mm256_cmpeq_epi32(vx1, vx2_2);
					vnewvalue = _mm256_max_epi32(vnewvalue, v2);
					vnewvalue = _mm256_blendv_epi8(vnewvalue, _mm256_add_epi32(v1, vone), mask);
					Input_Output_i_keep_src_5(out, v1, vnewvalue, in);
					Input_i_5(vx2_2,in_x2);

					mask = _mm256_cmpeq_epi32(vx1, vx2_3);
					vnewvalue = _mm256_max_epi32(vnewvalue, v3);
					vnewvalue = _mm256_blendv_epi8(vnewvalue, _mm256_add_epi32(v2, vone), mask);
					Input_Output_i_keep_src_6(out, v2, vnewvalue, in);
					Input_i_6(vx2_3,in_x2);

					mask = _mm256_cmpeq_epi32(vx1, vx2_1);
					vnewvalue = _mm256_max_epi32(vnewvalue, v4);
					vnewvalue = _mm256_blendv_epi8(vnewvalue, _mm256_add_epi32(v3, vone), mask);
					Input_Output_i_keep_src_7(out, v3, vnewvalue, in);
					Input_i_7(vx2_1,in_x2);

					mask = _mm256_cmpeq_epi32(vx1, vx2_2);
					vnewvalue = _mm256_max_epi32(vnewvalue, v1);
					vnewvalue = _mm256_blendv_epi8(vnewvalue, _mm256_add_epi32(v4, vone), mask);
					Input_Output_i_keep_src_8(out, v4, vnewvalue, in);
					Input_i_8(vx2_2,in_x2);

					vstorei(lcs[x - STRIDE * VECLEN_INT], out);

					mask = vx2_2;
					vx2_2 = vx2_1;
					vx2_1 = vx2_3;
					vx2_3 = mask;
				}

				vstorei(head[0][0],v1);
				vstorei(head[1][0],v2);
				vstorei(head[2][0],v3);
				vstorei(head[3][0],v4);

				xx = x - STRIDE;

				for (t = 1; t < VECLEN_INT; t++, xx -= STRIDE) {

					for(x = xx - STRIDE; x < xx; x++){

						lcs[x] = head[x - xx + STRIDE + 1][t];
					}
				}

				xx += STRIDE * VECLEN_INT;
				for (t = tt; t < tt + VECLEN_INT; t++, xx -= STRIDE) {

					lcs_t_minus_1_x_minus_1 = head[0][t - tt];
					
					for ( k = xx - STRIDE, x = k - LCSSTART + XSTART; x < myxbeg + xsize; x++, k++) {
						if ( st[t] == sx[x] ) lcs_t_x = 1 + lcs_t_minus_1_x_minus_1;
						else lcs_t_x = max(lcs[k-1], lcs[k]);
						lcs_t_minus_1_x_minus_1 = lcs[k];
						lcs[k] = lcs_t_x;
					}
					lcsleft[mytbeg / bt][t - mytbeg] = lcs_t_minus_1_x_minus_1; 
				}
			}


			for (; t < mytbeg + tsize; t++) {

				lcs_t_minus_1_x_minus_1 = lcsleft[mytbeg / bt][t - mytbeg];
				
				for ( x = myxbeg, k = myxbeg - XSTART + LCSSTART; x < myxbeg + xsize; x++, k++) {

					if (x == myxbeg) {
						lcs_t_x_minus_1 = lcsleft[mytbeg / bt][t - mytbeg + 1];
					} else {
						lcs_t_x_minus_1 = lcs[k - 1];
					}

					if ( st[t] == sx[x] ) {
						lcs_t_x = 1 + lcs_t_minus_1_x_minus_1;
					} else {
						lcs_t_x = max(lcs_t_x_minus_1, lcs[k]);
					} 

					lcs_t_minus_1_x_minus_1 = lcs[k];
					lcs[k] = lcs_t_x;
				}
				lcsleft[mytbeg / bt][t - mytbeg] = lcs_t_minus_1_x_minus_1;         
			}

			lcsleft[mytbeg / bt][t - mytbeg] = lcs[myxbeg + xsize - 1]; 
		}
	}
    free(lcsleft);
    return lcs[LCSSTART + nx - 1];
}
/*

			8|*
			7|777*
				6666*
				   5555*
					  4444*
					   	 3333*
							2222*
							   1111*
								  0000*
*/
/*

1111
1111
1111
1111
00001111
00001111
00001111
000011112222
        1111
        1111
        1111
        1111
*/
