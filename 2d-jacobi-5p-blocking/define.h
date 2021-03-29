#include "../common.h"


#define STRIDE 2

#define XSTART 1
#define YSTART 4
#define YSLOPE 1
#define XSLOPE 1

#define INIT 1.0 * (rand() % 1024)

typedef __m256d vec;


// #define simplestencil
#define Compute_one(v_center_3, v_center_0, v_center_1, v_center_2, a, b)	v_center_2 = loadv_2d_x_c_blk(y + a);\
																			vstore_2d_x_pp_blk(v_center_3, y + a - 1);\
																			Compute_1vector(v_center_0, v_center_1, v_center_2, \
																							loadv_2d_x_m_blk(y + a - 1), \
																							loadv_2d_x_p_blk(y + a - 1));\
																			Input_Output_##b(out, v_center_0, in)

#define Compute_lastone(v_center_2, v_center_3, v_center_0, v_center_1, a, b)	v_center_1 = ( y > myybeg - VECLEN + 1 + myyb - VECLEN - VECLEN * 2 ) ? \
																										setv_2d_blk(x, y + a)\
																									:	loadv_2d_x_c_blk(y + a) ;\
																				v_x_minus = loadv_2d_x_m_blk(y + a - 1);\
																				vstore_2d_x_pp_blk(v_center_2, y + a - 1);\
																				Compute_1vector(v_center_3, v_center_0, v_center_1, \
																								v_x_minus,\
																								loadv_2d_x_p_blk(y + a - 1)); \
																				Input_Output_##b(out, v_center_3, in)


#if defined(simplestencil)

			#define C0 0.2
			#define SET_COFF  vec vc0 = _mm256_set1_pd(C0)
			#define Compute_scalar(A, t, x, y) A[(t+1)%2][x][y] = C0 * (A[t%2][x][y] + A[t%2][x][y-1] + A[t%2][x+1][y] + A[t%2][x-1][y] + A[t%2][x][y+1])
			#define Compute_1vector_right(v_center_0, \
									v_center_1, \
									v_center_2, \
									v_x_minus_1, \
									v_x_plus_1) _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(\
																									v_center_1,\
																									v_center_0),\
																									v_x_plus_1),\
																									v_x_minus_1),\
																									v_center_2),vc0)
#elif defined(heat)

			#define C0 0.125
			#define C1 2.0
			#define SET_COFF 		__m256d vc0 = _mm256_set1_pd(C0);\
									__m256d vc1 = _mm256_set1_pd(C1)

			#define Compute_scalar(A, t, x, y) A[(t+1)%2][x][y] =	0.125 * (A[t%2][x+1][y] - 2.0 * A[t%2][x][y] + A[t%2][x-1][y]) + \
																	0.125 * (A[t%2][x][y+1] - 2.0 * A[t%2][x][y] + A[t%2][x][y-1]) + \
																	A[t%2][x][y]

			#define Compute_1vector_right(v_center_0, \
									v_center_1, \
									v_center_2, \
									v_x_minus_1, \
									v_x_plus_1) _mm256_add_pd(	_mm256_add_pd(\
																	_mm256_mul_pd(vc0, _mm256_add_pd(_mm256_fnmadd_pd(vc1, v_center_1, v_x_plus_1), v_x_minus_1)),\
																	_mm256_mul_pd(vc0, _mm256_add_pd(_mm256_fnmadd_pd(vc1, v_center_1, v_center_2), v_center_0))),\
																v_center_1)

#else


			#define C0 0.5
			#define C1 0.125
			#define SET_COFF 		__m256d vc0 = _mm256_set1_pd(C0);\
									__m256d vc1 = _mm256_set1_pd(C1)

			#define Compute_scalar(A, t, x, y) A[(t+1)%2][x][y] = 	C0 * A[t%2][x][y] + \
																	C1 * (A[t%2][x][y-1] + \
																			A[t%2][x+1][y] + \
																			A[t%2][x-1][y] + \
																			A[t%2][x][y+1])

			#define Compute_1vector_right(v_center_0, \
									v_center_1, \
									v_center_2, \
									v_x_minus_1, \
									v_x_plus_1) _mm256_fmadd_pd(v_center_1, vc0,\
													_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(\
																									v_center_0,\
																									v_x_plus_1),\
																									v_x_minus_1),\
																									v_center_2),vc1))

#endif

#define Compute_1vector(v_center_0, \
						v_center_1, \
						v_center_2, \
						v_x_minus_1, \
						v_x_plus_1) v_center_0=Compute_1vector_right(	v_center_0, \
																		v_center_1, \
																		v_center_2, \
																		v_x_minus_1, \
																		v_x_plus_1)

#define compute_5p_1vec(a, b, c, d, e) e = Compute_1vector_right(b, a, d, e, c)


void vectime(double * B, int NX, int NY, int T, int xb, int yb, int tb);
void naive_vector(double * B, int NX, int NY, int T, int xb, int yb, int tb);
int checkresult( int NX, int NY, double (* A)[ NY+2*YSTART], double (* B)[ NY+2*YSTART] );

