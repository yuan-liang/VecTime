#include "defines.h"

void vectime(double * B, int NX, int NY, int NZ, int T){

	int x, y, z;
	int t = 0, tt, xx;
	__m256d vcenter;
	__m256d vx_minus_1, vx_plus_1;
	__m256d vy_minus_1, vy_plus_1;
	__m256d vz_minus_1, vz_plus_1;
	__m256d vmiddlein, vmiddleout;
	SET_COFF;
	double tmp[4];
	double (* A)[ NY+2*YSTART][NZ+2*ZSTART] =  (double (*)[ NY+2*YSTART][NZ+2*ZSTART])  B;


	for ( tt = 0; tt <= T - VECLEN; tt += VECLEN){	

		//		4|00000000|
		//		4|43322110|
		for( t = tt ; t < tt + VECLEN ; t++){
			for ( x = XSTART; x < XSTART + STRIDE * (VECLEN - (t - tt)) - 1; x++){
				for ( y = YSTART; y < NY + YSTART; y++){
					for (z = ZSTART; z < NZ + ZSTART; z++){
						Compute_scalar(A, x, y, z);
					}	
				}
			}
		}

		//		4|43322110|0
		//		 |#*#*#*#*|0
		for(x = XSTART - 1 + STRIDE * VECLEN - 1; x < XSTART - 1 + STRIDE * VECLEN + 1; x++){
			for ( y = YSTART - YSLOPE; y <= NY + YSTART; y++){
				for ( z = ZSTART ; z <= NZ + ZSTART - VECLEN; z += VECLEN){	
					vcenter 	= _mm256_loadu_pd(&A[x - STRIDE * 0][y][z]);
					vx_plus_1	= _mm256_loadu_pd(&A[x - STRIDE * 1][y][z]);
					vy_plus_1 	= _mm256_loadu_pd(&A[x - STRIDE * 2][y][z]);
					vz_plus_1 	= _mm256_loadu_pd(&A[x - STRIDE * 3][y][z]);
					transpose(vcenter, vx_plus_1, vy_plus_1, vz_plus_1, vmiddlein, vmiddleout);
					_mm256_storeu_pd(&A[x - STRIDE * 0][y][z], vcenter);
					_mm256_storeu_pd(&A[x - STRIDE * 1][y][z], vx_plus_1);
					_mm256_storeu_pd(&A[x - STRIDE * 2][y][z], vy_plus_1);
					_mm256_storeu_pd(&A[x - STRIDE * 3][y][z], vz_plus_1);
					
				}
			}
		}
	



		for (x = XSTART + STRIDE * VECLEN - 1; x < XSTART + NX; x++){

		//		4|43322110|0
		//		4|4332211x|0
		//		 |0#*#*#*#|*
			y = YSTART - YSLOPE;
			for ( z = ZSTART ; z <= NZ + ZSTART - VECLEN; z += VECLEN){	

				vmiddlein 	= _mm256_loadu_pd(&A[x + 1][y][z]);

				vcenter 	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 0][y][z]);
				vx_plus_1	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 1][y][z]);
				vy_plus_1	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 2][y][z]);
				vz_plus_1	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 3][y][z]);

				Input_Output_1(vmiddleout,vcenter,vmiddlein);
				Input_Output_2(vmiddleout,vx_plus_1,vmiddlein);
				Input_Output_3(vmiddleout,vy_plus_1,vmiddlein);
				Input_Output_4(vmiddleout,vz_plus_1,vmiddlein);

				_mm256_storeu_pd(&A[x - 1 - STRIDE * 3][y][z], vmiddleout);

				_mm256_storeu_pd(&A[x + 1 - STRIDE * 0][y][z], vcenter);
				_mm256_storeu_pd(&A[x + 1 - STRIDE * 1][y][z], vx_plus_1);
				_mm256_storeu_pd(&A[x + 1 - STRIDE * 2][y][z], vy_plus_1);
				_mm256_storeu_pd(&A[x + 1 - STRIDE * 3][y][z], vz_plus_1);				
			}



			for(y = YSTART; y < YSTART + NY ; y++){
	
				vz_minus_1 = _mm256_set_pd( A[x - STRIDE * 3]	[y] [ZSTART - ZSLOPE],\
											A[x - STRIDE * 2]	[y] [ZSTART - ZSLOPE], \
											A[x - STRIDE * 1] 	[y] [ZSTART - ZSLOPE], \
											A[x - STRIDE * 0]	[y] [ZSTART - ZSLOPE]); 

				vcenter 	= _mm256_loadu_pd(&A[x][y][ZSTART]);
				
	
		//		4|43322110|0
		//		4|4332211x|0
		//		 |0#*#*#*#|*
				for ( z = ZSTART ; z <= NZ + ZSTART - VECLEN; z += VECLEN){	

					vmiddlein 	= _mm256_loadu_pd(&A[x + 1]	[y]		[z]);


					vz_plus_1 = _mm256_loadu_pd(&A[x - STRIDE][y][z]);

					vx_minus_1 	= _mm256_loadu_pd(&A[x - 1][y][z]);
					Input_Output_keep_src_1(vmiddleout, vx_plus_1, vx_minus_1, vmiddlein);
					_mm256_storeu_pd(&A[x + 1][y][z], vx_plus_1);

					vy_plus_1 = _mm256_loadu_pd(&A[x][y + 1][z]);
					vy_minus_1 = _mm256_loadu_pd(&A[x][y - 1][z]);	

					Compute_1vector(    vcenter,\
										vz_minus_1,vz_plus_1, \
										vx_minus_1, vx_plus_1,\
										vy_minus_1, vy_plus_1);
										
					_mm256_storeu_pd(&A[x][y][z], vz_minus_1);
					vcenter = vz_plus_1;



					vz_plus_1 = _mm256_loadu_pd(&A[x - STRIDE * 2][y][z]);	

		
					vx_minus_1 	= _mm256_loadu_pd(&A[x - 1 - STRIDE][y][z]);
					Input_Output_keep_src_2(vmiddleout, vx_plus_1, vx_minus_1, vmiddlein);
					_mm256_storeu_pd(&A[x + 1 - STRIDE * 1][y][z], vx_plus_1);

					vy_plus_1 = _mm256_loadu_pd(&A[x - STRIDE][y + 1][z]);
					vy_minus_1 = _mm256_loadu_pd(&A[x - STRIDE][y - 1][z]);	

					Compute_1vector(    vcenter,\
										vz_minus_1,vz_plus_1, \
										vx_minus_1, vx_plus_1,\
										vy_minus_1, vy_plus_1);
										
					_mm256_storeu_pd(&A[x - STRIDE][y][z], vz_minus_1);
					vcenter = vz_plus_1;




					vz_plus_1 = _mm256_loadu_pd(&A[x - STRIDE * 3][y][z]);	

					vx_minus_1 	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 2][y][z]);
					Input_Output_keep_src_3(vmiddleout, vx_plus_1, vx_minus_1, vmiddlein);
					_mm256_storeu_pd(&A[x + 1 - STRIDE * 2][y][z], vx_plus_1);

					vy_plus_1 = _mm256_loadu_pd(&A[x - STRIDE * 2][y + 1][z]);
					vy_minus_1 = _mm256_loadu_pd(&A[x - STRIDE * 2][y - 1][z]);	

					Compute_1vector(    vcenter,\
										vz_minus_1,vz_plus_1, \
										vx_minus_1, vx_plus_1,\
										vy_minus_1, vy_plus_1);
										
					_mm256_storeu_pd(&A[x - STRIDE * 2][y][z], vz_minus_1);
					vcenter = vz_plus_1;


					if(z + VECLEN <= NZ + ZSTART - VECLEN){
						vz_plus_1 = _mm256_loadu_pd(&A[x][y][z + VECLEN]);
					}else{
						vz_plus_1 = _mm256_set_pd(	A[x - STRIDE * 3]	[y] [z + VECLEN],\
													A[x - STRIDE * 2]	[y] [z + VECLEN], \
													A[x - STRIDE]	 	[y] [z + VECLEN], \
													A[x]			 	[y] [z + VECLEN]); 
					}


					vx_minus_1 	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 3][y][z]);
					Input_Output_keep_src_4(vmiddleout, vx_plus_1, vx_minus_1, vmiddlein);
					_mm256_storeu_pd(&A[x + 1 - STRIDE * 3][y][z], vx_plus_1);
					_mm256_storeu_pd(&A[x + 1 - STRIDE * 4][y][z], vmiddleout);

					vy_plus_1 = _mm256_loadu_pd(&A[x - STRIDE * 3][y + 1][z]);
					vy_minus_1 = _mm256_loadu_pd(&A[x - STRIDE * 3][y - 1][z]);	

					Compute_1vector(    vcenter,\
										vz_minus_1,vz_plus_1, \
										vx_minus_1, vx_plus_1,\
										vy_minus_1, vy_plus_1);
										
					_mm256_storeu_pd(&A[x - STRIDE * 3][y][z], vz_minus_1);
					vcenter = vz_plus_1;
				}
				for(; z < ZSTART + NZ; z++){

					vx_minus_1 = _mm256_set_pd( A[x - 1 - STRIDE * 3][y] [z],\
												A[x - 1 - STRIDE * 2][y] [z], \
												A[x - 1 - STRIDE]	 [y] [z], \
												A[x - 1]			 [y] [z]); 

					vx_plus_1 = _mm256_set_pd( 	A[x + 1 - STRIDE * 3][y] [z],\
												A[x + 1 - STRIDE * 2][y] [z], \
												A[x + 1 - STRIDE]	 [y] [z], \
												A[x + 1]			 [y] [z]); 

					vy_minus_1 = _mm256_set_pd( A[x - STRIDE * 3][y - 1] [z],\
												A[x - STRIDE * 2][y - 1] [z], \
												A[x - STRIDE]	 [y - 1] [z], \
												A[x]			 [y - 1] [z]); 

					vy_plus_1 = _mm256_set_pd(  A[x - STRIDE * 3][y + 1] [z],\
												A[x - STRIDE * 2][y + 1] [z], \
												A[x - STRIDE]	 [y + 1] [z], \
												A[x]			 [y + 1] [z]); 	

					vz_plus_1 = _mm256_set_pd(  A[x - STRIDE * 3][y] [z + 1],\
												A[x - STRIDE * 2][y] [z + 1], \
												A[x - STRIDE]	 [y] [z + 1], \
												A[x]			 [y] [z + 1]);

					Compute_1vector(    vcenter,\
										vz_minus_1,vz_plus_1, \
										vx_minus_1, vx_plus_1,\
										vy_minus_1, vy_plus_1);

					_mm256_storeu_pd(tmp, vz_minus_1);

					A[x]   			 [y][z] = tmp[0];
					A[x - STRIDE]	 [y][z] = tmp[1];
					A[x - STRIDE * 2][y][z] = tmp[2];
					A[x - STRIDE * 3][y][z] = tmp[3];

					vcenter = vz_plus_1;
				}
			}


			y = YSTART + NY;
			for ( z = ZSTART ; z <= NZ + ZSTART - VECLEN; z += VECLEN){	

				vmiddlein 	= _mm256_loadu_pd(&A[x + 1][y][z]);

				vcenter 	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 0][y][z]);
				vx_plus_1	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 1][y][z]);
				vy_plus_1	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 2][y][z]);
				vz_plus_1	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 3][y][z]);

				Input_Output_1(vmiddleout,vcenter,vmiddlein);
				Input_Output_2(vmiddleout,vx_plus_1,vmiddlein);
				Input_Output_3(vmiddleout,vy_plus_1,vmiddlein);
				Input_Output_4(vmiddleout,vz_plus_1,vmiddlein);

				_mm256_storeu_pd(&A[x - 1 - STRIDE * 3][y][z], vmiddleout);

				_mm256_storeu_pd(&A[x + 1 - STRIDE * 0][y][z], vcenter);
				_mm256_storeu_pd(&A[x + 1 - STRIDE * 1][y][z], vx_plus_1);
				_mm256_storeu_pd(&A[x + 1 - STRIDE * 2][y][z], vy_plus_1);
				_mm256_storeu_pd(&A[x + 1 - STRIDE * 3][y][z], vz_plus_1);			
			}
		}
			
		for(; x < XSTART + NX + STRIDE; x++){
			for ( y = YSTART - YSLOPE; y <= NY + YSTART; y++){
				for ( z = ZSTART ; z <= NZ + ZSTART - VECLEN; z += VECLEN){	
					vcenter 	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 0][y][z]);
					vx_plus_1	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 1][y][z]);
					vy_plus_1 	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 2][y][z]);
					vz_plus_1 	= _mm256_loadu_pd(&A[x - 1 - STRIDE * 3][y][z]);
					transpose(vcenter, vx_plus_1, vy_plus_1, vz_plus_1, vmiddlein, vmiddleout);
					_mm256_storeu_pd(&A[x - 1 - STRIDE * 0][y][z], vcenter);
					_mm256_storeu_pd(&A[x - 1 - STRIDE * 1][y][z], vx_plus_1);
					_mm256_storeu_pd(&A[x - 1 - STRIDE * 2][y][z], vy_plus_1);
					_mm256_storeu_pd(&A[x - 1 - STRIDE * 3][y][z], vz_plus_1);
					
				}
			}
		}
		for(t = tt ; t < tt + VECLEN - 1; t++){	
			for ( x = NX + XSTART - STRIDE * (t - tt + 1); x < NX + XSTART; x ++){
				for (y = YSTART; y < NY + YSTART; y++) {
					for (z = ZSTART; z < NZ + ZSTART; z++){
						Compute_scalar(A,x,y,z);
					}
				}
			}
		}
		t++;
	}	
	for (; t < T; t++) {
		for (x = XSTART; x < NX + XSTART; x++) {
			for (y = YSTART; y < NY + YSTART; y++) {
				for (z = ZSTART; z < NZ + ZSTART; z++){
					Compute_scalar(A,x,y,z);
				}
			}
		}
	}
}