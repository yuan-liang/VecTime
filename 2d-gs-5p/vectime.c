#include "define.h"

void vectime(double * B, int NX, int NY, int T){

	int x, y, t = 0, tt, xx;
	__m256d v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11;
	__m256d out, in_x;
	SET_COFF;
	double tmp[4];
	double (* A)[ NY + 2 * YSTART] =  (double (*)[ NY + 2 * YSTART])  B;

	for ( tt = 0; tt <= T - VECLEN; tt += VECLEN){	

		//0|00000000
		//0|43322110
		for( t = tt ; t < tt + VECLEN; t++){
			for ( x = XSTART; x < XSTART + STRIDE * (VECLEN - (t - tt)) - 1; x++){
				for ( y = YSTART; y < YSTART + NY ; y++){
					Compute_scalar(A, x, y);	
				}
			}
		}

		for(x = XSTART + STRIDE * (VECLEN - 1); x < XSTART + STRIDE * (VECLEN - 1) + 2; x++){
			for ( y = YSTART; y <= YSTART + NY - VECLEN; y += VECLEN ){
				v1 = _mm256_loadu_pd(&A[x - STRIDE * 0][y]);
				v2 = _mm256_loadu_pd(&A[x - STRIDE * 1][y]);
				v3 = _mm256_loadu_pd(&A[x - STRIDE * 2][y]);
				v4 = _mm256_loadu_pd(&A[x - STRIDE * 3][y]);
				transpose(v1, v2, v3, v4, v5, v6);
				_mm256_storeu_pd(&A[x - STRIDE * 0][y], v1);
				_mm256_storeu_pd(&A[x - STRIDE * 1][y], v2);
				_mm256_storeu_pd(&A[x - STRIDE * 2][y], v3);
				_mm256_storeu_pd(&A[x - STRIDE * 3][y], v4);
			}
		}
		//0|43322110
		//0|#*#*#*#*
		//          x
		for (; x <= XSTART + NX; x++){
			
			v1 = _mm256_set_pd( A[x - 1 - STRIDE * 3]	 [YSTART - YSLOPE], \
								A[x - 1 - STRIDE * 2]	 [YSTART - YSLOPE], \
								A[x - 1 - STRIDE]	 	 [YSTART - YSLOPE], \
								A[x - 1]			 	 [YSTART - YSLOPE]); 

			v4		= load_vcenter_1(x, YSTART);
			in_x	= _mm256_loadu_pd(&A[x][YSTART]);
	
			for ( y = YSTART ; y <= NY + YSTART - VECLEN; y += VECLEN){	

				v3 = load_vx_minus_1_1(x, y);
				Input_Output_keep_src_1(out, v5, v3, in_x);
				v7 = load_vcenter_2(x, y);				
				Compute_1vector(    v3, \
								v1, v4, v7, \
								    v5);
				store_vx_plus_1_1(v5, x, y);
				store_newvalue_1(v4, x, y);



				v6 = load_vx_minus_1_2(x, y);
				Input_Output_keep_src_2(out, v8, v6, in_x);
				v10 = load_vcenter_3(x, y);
				Compute_1vector(    v6, \
								v4, v7, v10, \
								    v8);
				store_vx_plus_1_2(v8, x, y);
				store_newvalue_2(v7, x, y);


				
				v9 = load_vx_minus_1_3(x, y);
				Input_Output_keep_src_3(out, v11, v9, in_x);
				v1 = load_vcenter_4(x, y);
				Compute_1vector(    v9, \
								v7, v10, v1, \
								    v11);
				store_vx_plus_1_3(v11, x, y);
				store_newvalue_3(v10, x, y);

				
				v0 = load_vx_minus_1_4(x, y);
				Input_Output_keep_src_4(out, v2, v0, in_x);
				if(y + VECLEN <= NY + YSTART - VECLEN){
					in_x 		= _mm256_loadu_pd(&A[x][y + VECLEN]);
					v4 = load_vcenter_1(x, y + VECLEN);
				}else{
					v4 = _mm256_set_pd( A[x - 1 - STRIDE * 3]	 [y + VECLEN], \
										A[x - 1 - STRIDE * 2]	 [y + VECLEN], \
										A[x - 1 - STRIDE]	 	 [y + VECLEN], \
										A[x - 1]			 	 [y + VECLEN]); 
				}
				Compute_1vector(     v0, \
								v10, v1, v4, \
								     v2);
				store_vx_plus_1_4(v2, x, y);
				_mm256_storeu_pd(&A[x - STRIDE * 4][y], out);
				store_newvalue_4(v1, x, y);

			}


			
			for(y += 1; y <= YSTART + NY; y++){

				v3 = _mm256_set_pd( A[x - 2 - STRIDE * 3]	 [y - 1], \
									A[x - 2 - STRIDE * 2]	 [y - 1], \
									A[x - 2 - STRIDE]	 	 [y - 1], \
									A[x - 2]			 	 [y - 1]); 
				v7 = _mm256_set_pd( A[x - 1 - STRIDE * 3]	 [y], \
									A[x - 1 - STRIDE * 2]	 [y], \
									A[x - 1 - STRIDE]	 	 [y], \
									A[x - 1]			 	 [y]); 
				v5 = _mm256_set_pd( A[x - STRIDE * 3]	 [y - 1], \
									A[x - STRIDE * 2]	 [y - 1], \
									A[x - STRIDE]	 	 [y - 1], \
									A[x]			 	 [y - 1]);

				Compute_1vector(    v3, \
								v1, v4, v7, \
								    v5);
				_mm256_storeu_pd(tmp, v4);

				A[x - 1]   			 [y - 1] = tmp[0];
				A[x - 1 - STRIDE]	 [y - 1] = tmp[1];
				A[x - 1 - STRIDE * 2][y - 1] = tmp[2];
				A[x - 1 - STRIDE * 3][y - 1] = tmp[3];
				
				v1 = v4;
				v4 = v7;
			}
		} 

		xx = x;
		for(x = xx - STRIDE; x < xx; x++){
			for ( y = YSTART; y <= YSTART + NY - VECLEN; y += VECLEN ){
				v1 = _mm256_loadu_pd(&A[x][y]);
				v2 = _mm256_loadu_pd(&A[x - STRIDE][y]);
				v3 = _mm256_loadu_pd(&A[x - STRIDE * 2][y]);
				v4 = _mm256_loadu_pd(&A[x - STRIDE * 3][y]);
				transpose(v1, v2, v3, v4, v5, v6);
				_mm256_storeu_pd(&A[x][y], v1);
				_mm256_storeu_pd(&A[x - STRIDE][y], v2);
				_mm256_storeu_pd(&A[x - STRIDE * 2][y], v3);
				_mm256_storeu_pd(&A[x - STRIDE * 3][y], v4);
			}
		}
		xx -= 1;
		for(t = tt ; t < tt + VECLEN; t++, xx -= STRIDE){	
			for ( x = xx; x < NX + XSTART; x ++){
				for (y = YSTART; y < NY + YSTART; y++) {
					Compute_scalar(A, x, y);
				}
			}
		}
	}
	for ( ; t < T; t++) {
		for (x = XSTART; x < NX + XSTART; x++) {
			for (y = YSTART; y < NY + YSTART; y++) {
				Compute_scalar(A, x, y);
			}
		}
	}
}