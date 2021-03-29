#include "../common.h"

#define STRIDE 2
#define LANESTRIDE 1

#define XSTART 1
#define YSTART 4
#define YSLOPE 1
#define XSLOPE 1

#define INIT 1.0 * (rand() % 1024)


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
			#define C1 -2.0
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
																	_mm256_mul_pd(vc0, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(vc1, v_center_1), v_x_plus_1), v_x_minus_1)),\
																	_mm256_mul_pd(vc0, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(vc1, v_center_1), v_center_2), v_center_0))),\
																v_center_1)
												//_mm256_add_pd(	_mm256_add_pd(\
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






void vectime(double* A, int NX, int NY, int T);
void naive_scalar(double * A, int NX, int NY, int T);
void naive_vector(double * A, int NX, int NY, int T);
int checkresult( int NX, int NY, double (* A_correct)[ NY+2*YSTART], double (* A)[ NY+2*YSTART]);


typedef __m256d vec;

