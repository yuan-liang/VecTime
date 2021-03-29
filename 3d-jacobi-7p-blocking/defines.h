#include "../common.h"


#define STRIDE 2

#define XSTART 1
#define YSTART 1
#define ZSTART 4

#define YSLOPE 1
#define XSLOPE 1
#define ZSLOPE 1

#define myxb_threshold  2*VECLEN
#define myyb_threshold	VECLEN
#define myzb_threshold	VECLEN

#define INIT 1.0 * (rand() % 1024)

typedef __m256d vec;


#define Compute_one(v_center_0, v_center_1, v_center_2, a, b)   v_center_2 = loadv_3d_x_c_blk(y, z + a);\
                                                                v_y_minus = ( y == myybeg - VECLEN + 1 ) ? \
                                                                                load_v(x, y, z, 0, -1, a - 1)\
                                                                                : loadv_3d_x_c_blk(y - 1, z + a - 1);\
                                                                v_y_plus  = ( y == myybeg - VECLEN + 1 + myyb - 1 ) ? \
                                                                                load_v(x, y, z, 0, 1, a - 1) \
                                                                                : loadv_3d_x_c_blk(y + 1, z + a -1); \
                                                                v_x_minus = loadv_3d_x_m_blk(y, z + a - 1);\
                                                                v_x_plus = loadv_3d_x_p_blk(y, z + a - 1);\
                                                                Compute_1vector(v_center_1, v_center_0, v_center_2, v_x_minus, \
                                                                                v_x_plus, v_y_minus, v_y_plus);\
                                                                if(y < myybeg - VECLEN + 1 + myyb - 1) {\
                                                                    Input_Output_##b(out, v_center_0, in);\
                                                                    store_x_pp(v_center_0, x, y, z, 1, a);\
                                                                }\
                                                                else {  \
                                                                    store_v(v_center_0, x, y, z, 0, 0, a - 1);\
                                                                }



#define  Compute_lastone(v_center_3, v_center_0, v_center_1, a, b)  v_center_1 =  (z > myzbeg - VECLEN + 1 + myzb - VECLEN - VECLEN * 2 ) ? \
                                                            load_v(x, y, z, 0, 0, a)\
                                                        :   loadv_3d_x_c_blk(y, z + a);\
                                            v_y_minus = ( y == myybeg - VECLEN + 1 ) ? \
                                                            load_v(x, y, z, 0, -1, a - 1) \
                                                            : loadv_3d_x_c_blk(y - 1, z + a - 1);\
                                            v_y_plus  = ( y == myybeg - VECLEN + 1 + myyb - 1 ) ? \
                                                            load_v(x, y, z, 0, 1, a - 1)\
                                                            : loadv_3d_x_c_blk(y + 1, z + a - 1);\
                                            v_x_minus = loadv_3d_x_m_blk(y, z + a - 1);\
                                            v_x_plus = loadv_3d_x_p_blk(y, z + a - 1);  \
                                            Compute_1vector(v_center_0, v_center_3, v_center_1, v_x_minus, \
                                                            v_x_plus, v_y_minus, v_y_plus); \
                                            if(y < myybeg - VECLEN + 1 + myyb - 1) {\
                                                if (z >  myzbeg - VECLEN + 1 + myzb - VECLEN - VECLEN * 2) {\
									            	store_v(v_center_3, x, y, z, 0, 0, a - 1);\
									            }\
                                                Input_Output_##b(out, v_center_3, in);	\
                                            }\
                                            else{\
                                                store_v(v_center_3, x, y, z, 0, 0, a - 1);\
                                            }

#if defined(simplestencil)

		#define C1 0.143

		#define SET_COFF  vec c1;\
						vallset(c1,C1)

		#define Compute_scalar(A,t,i,j,k) A[(t+1)%2][i][j][k] =  C1 * ((((((A[t%2][i][j][k] + A[t%2][i][j][k-1]) \
																	+ A[t%2][i][j][k+1]) + A[t%2][i][j-1][k]) \
																	+ A[t%2][i][j+1][k]) + A[t%2][i-1][j][k]) \
																	+ A[t%2][i+1][j][k])

		#define Compute_1vector(vcenter,vz_minus_1,vz_plus_1,vy_minus_1, vy_plus_1,vx_minus_1, vx_plus_1)	vz_minus_1 = _mm256_mul_pd(\
													_mm256_add_pd(\
													_mm256_add_pd(\
													_mm256_add_pd(\
													_mm256_add_pd(\
													_mm256_add_pd(\
													_mm256_add_pd(vcenter,vz_minus_1),vz_plus_1),vy_minus_1),vy_plus_1),vx_minus_1),vx_plus_1),c1);

#elif defined(heat)

		#define C0 0.125
		#define C1 -2.0
		#define SET_COFF 		__m256d vc0 = _mm256_set1_pd(C0);\
								__m256d vc1 = _mm256_set1_pd(C1)

		#define Compute_scalar(A,t,x,y,z)	A[(t+1)%2][x][y][z] =	  0.125*(A[t%2][x - 1][y][z] - 2.0 * A[t%2][x][y][z] + A[t%2][x + 1][y][z])\
																	+ 0.125*(A[t%2][x][y - 1][z] - 2.0 * A[t%2][x][y][z] + A[t%2][x][y + 1][z])\
																	+ 0.125*(A[t%2][x][y][z - 1] - 2.0 * A[t%2][x][y][z] + A[t%2][x][y][z + 1])\
																	+ A[t%2][x][y][z]

		#define Compute_1vector(vcenter,\
								vz_minus_1, vz_plus_1,\
								vy_minus_1, vy_plus_1,\
								vx_minus_1, vx_plus_1) vz_minus_1 = _mm256_add_pd( _mm256_add_pd( _mm256_add_pd(\
																_mm256_mul_pd(vc0, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(vc1, vcenter), vx_minus_1), vx_plus_1)),\
																_mm256_mul_pd(vc0, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(vc1, vcenter), vy_minus_1), vy_plus_1))),\
																_mm256_mul_pd(vc0, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(vc1, vcenter), vz_minus_1), vz_plus_1))),\
																vcenter)
														//vz_minus_1 = _mm256_add_pd( _mm256_add_pd( _mm256_add_pd(\
																_mm256_mul_pd(vc0, _mm256_add_pd(_mm256_fnmadd_pd(vc1, vcenter, vx_minus_1), vx_plus_1)),\
																_mm256_mul_pd(vc0, _mm256_add_pd(_mm256_fnmadd_pd(vc1, vcenter, vy_minus_1), vy_plus_1))),\
																_mm256_mul_pd(vc0, _mm256_add_pd(_mm256_fnmadd_pd(vc1, vcenter, vz_minus_1), vz_plus_1))),\
																vcenter)
#else

#define C0 0.4
#define C1 0.1

#define SET_COFF  vec vc1, vc0 ;\
				 vallset(vc1,C1);\
				 vallset(vc0,C0)

#define Compute_scalar(A,t,i,j,k) A[(t+1)%2][i][j][k] =  C0 * A[t%2][i][j][k] + C1 * (((((( (A[t%2][i][j][k-1]) \
															 + A[t%2][i][j][k+1])) + A[t%2][i][j-1][k]) \
															 + A[t%2][i][j+1][k]) + A[t%2][i-1][j][k]) \
															 + A[t%2][i+1][j][k])

#define Compute_1vector(vcenter,vz_minus_1,vz_plus_1,vy_minus_1, vy_plus_1,vx_minus_1, vx_plus_1)	vz_minus_1 = _mm256_fmadd_pd(vcenter, vc0,\
                                                                                        _mm256_mul_pd(\
                                                                                        _mm256_add_pd(\
                                                                                        _mm256_add_pd(\
                                                                                        _mm256_add_pd(\
                                                                                        _mm256_add_pd(\
                                                                                        _mm256_add_pd(vz_minus_1,vz_plus_1),vy_minus_1),vy_plus_1),vx_minus_1),vx_plus_1),vc1));

#endif



void naive_vec(double * A, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb);
void blocking_parallel_rectangle_scalar(double * B, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb);
void blocking_parallel_rectangle_vector(double * B, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb);
void blocking_parallel_rectangle_vectime(double * B, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb);
void blocking_parallel_rectangle_vectime_extra_array(double *A, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb);
void blocking_parallel_rectangle_vectime_extra_array_unroll8(double *A, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb);
int checkresult( int NX, int NY, int NZ, double (* A_correct)[ NY+2*YSTART][NZ+2*ZSTART], double (* A)[ NY+2*YSTART][NZ+2*ZSTART] );
void print256_vec(__m256d var, char str[] );
