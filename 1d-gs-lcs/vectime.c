#include "define.h"

int vectime(char * x1, char * x2, int nx1, int nx2, int * lcs){

	int tt, xx;
	int t = XSTART, x, k;
	int lcs_t_minus_1_x_minus_1;
	int lcs_t_x;
	int head[STRIDE + 1][VECLEN_INT];

	__m256i v1, v2, v3, v4;
	__m256i vnewvalue, in, out;
	__m256i	vx1, vx2_1, vx2_2, vx2_3, in_x2;
	__m256i vrotatei_high2low, vone, mask;


	for ( tt = XSTART; tt <= nx1 + XSTART - VECLEN_INT; tt += VECLEN_INT)
	{

		for(k = LCSSTART + STRIDE * (VECLEN_INT - 1) - 1; k <  LCSSTART + STRIDE * VECLEN_INT; k++){
			head[k - (LCSSTART + STRIDE * (VECLEN_INT - 1) - 1)][0] = lcs[k];
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


		for (t = tt; t < tt + VECLEN_INT - 1; t++) {
			lcs_t_minus_1_x_minus_1 = 0;
			k = LCSSTART;
			for ( x = XSTART; x < XSTART + STRIDE * (VECLEN_INT - (t - tt + 1)); x++, k++) {
				if ( x1[t] == x2[x] ) lcs_t_x = 1 + lcs_t_minus_1_x_minus_1;
				else lcs_t_x = max(lcs[k-1], lcs[k]);
				lcs_t_minus_1_x_minus_1 = lcs[k];
				lcs[k] = lcs_t_x;
			}

			for(x = k - STRIDE - 1; x < k; x++){
				head[x - (k - STRIDE - 1)][t - tt + 1] = lcs[x];
			}
		}

		v1 = vloadi(head[0][0]);
		v2 = vloadi(head[1][0]);
		v3 = vloadi(head[2][0]);
		v4 = vloadi(head[3][0]);

		int one[VECLEN_INT] = {1, 1, 1, 1, 1, 1, 1, 1};
		vone = vloadi(one[0]);

		head[STRIDE][0] = 0;
		vnewvalue = vloadi(head[STRIDE][0]);
		int rotatei_low2high[VECLEN_INT] = {1, 2, 3, 4, 5, 6, 7, 0};
		vrotatei_high2low = vloadi(rotatei_low2high[0]);
		vnewvalue = vrotate_high2lowi(vnewvalue);

		int rotatei_high2low[VECLEN_INT] = {7, 0, 1, 2, 3, 4, 5, 6};
		vrotatei_high2low = vloadi(rotatei_high2low[0]);

		int headx2[STRIDE][VECLEN_INT];

		for ( t = 0; t < STRIDE; t++){

			for ( x = 0; x < VECLEN_INT; x++){

				headx2[t][VECLEN_INT - 1 - x] = x2[XSTART + t + STRIDE * x];
			}			
		}

		vx2_1 = vloadi(headx2[0]);
		vx2_2 = vloadi(headx2[1]);
		vx2_3 = vloadi(headx2[2]);

		t = tt;
		vx1 = _mm256_setr_epi32(		(int) x1[t + 0], \
										(int) x1[t + 1], \
										(int) x1[t + 2], \
										(int) x1[t + 3], \
										(int) x1[t + 4], \
										(int) x1[t + 5], \
										(int) x1[t + 6], \
										(int) x1[t + 7]);

		for(x = LCSSTART + STRIDE * VECLEN_INT; x <= LCSSTART + nx2 - VECLEN_INT; x += VECLEN_INT){

			in = vloadi(lcs[x]);

			in_x2 =  _mm256_setr_epi32(	(int) x2[x + 0 - LCSSTART + XSTART], \
										(int) x2[x + 1 - LCSSTART + XSTART], \
										(int) x2[x + 2 - LCSSTART + XSTART], \
										(int) x2[x + 3 - LCSSTART + XSTART], \
										(int) x2[x + 4 - LCSSTART + XSTART], \
										(int) x2[x + 5 - LCSSTART + XSTART], \
										(int) x2[x + 6 - LCSSTART + XSTART], \
										(int) x2[x + 7 - LCSSTART + XSTART]);
										
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
			
			for ( k = xx - STRIDE, x = k - LCSSTART + XSTART; x < XSTART + nx2; x++, k++) {
				if ( x1[t] == x2[x] ) lcs_t_x = 1 + lcs_t_minus_1_x_minus_1;
				else lcs_t_x = max(lcs[k-1], lcs[k]);
				lcs_t_minus_1_x_minus_1 = lcs[k];
				lcs[k] = lcs_t_x;
			}
		}
	}

    for (; t < nx1 + XSTART; t++) {
        lcs_t_minus_1_x_minus_1 = 0;
        for ( k = LCSSTART, x = XSTART; x < nx2 + XSTART; x++, k++) {
            if ( x1[t] == x2[x] ) lcs_t_x = 1 + lcs_t_minus_1_x_minus_1;
            else lcs_t_x = max(lcs[k-1], lcs[k]);
			lcs_t_minus_1_x_minus_1 = lcs[k];
			lcs[k] = lcs_t_x;
        }          
    }

	return lcs[LCSSTART + nx2 - 1];
}

