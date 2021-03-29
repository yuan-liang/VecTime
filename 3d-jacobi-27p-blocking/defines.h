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

#define Add_4_vectors(a, b, c, d)	_mm256_add_pd(\
									_mm256_add_pd(\
									_mm256_add_pd(a, b), c), d)\

#define Add_4_d_1(x, y, z, a) Add_4_vectors(    load_x_m_blocking(x, y, z, 0, a),\
                                                load_x_p_blocking(x, y, z, 0, a),\
                                                ( y == myybeg - VECLEN + 1 + myyb - 1 ) ? load_v(x, y, z, 0, 1, a) : load_x_c_blocking(x, y, z, 1, a),\
                                                ( y == myybeg - VECLEN + 1 ) ? load_v(x, y, z, 0, -1, a) : load_x_c_blocking(x, y, z, -1, a) )

#define Add_4_d_2(x, y, z, a) Add_4_vectors(    ( y == myybeg - VECLEN + 1 + myyb - 1 ) ? load_v(x, y, z, -1, 1, a) : load_x_m_blocking(x, y, z, 1, a),\
                                                ( y == myybeg - VECLEN + 1 ) ? load_v(x, y, z, -1, -1, a) : load_x_m_blocking(x, y, z, -1, a),\
                                                ( y == myybeg - VECLEN + 1 + myyb - 1 ) ? load_v(x, y, z, 1, 1, a) : load_x_p_blocking(x, y, z, 1, a),\
                                                ( y == myybeg - VECLEN + 1 ) ? load_v(x, y, z, 1, -1, a) : load_x_p_blocking(x, y, z, -1, a) )


#ifdef simplestencil


#define C1 0.037


#define SET_COFF  __m256d vc1;\
				 vallset(vc1,C1);

#define Compute_scalar(A,t,i,j,k) A[(t+1)%2][i][j][k] =  C1 * (((((((((\
																A[t%2][i]	[j]		[k]  \
															+	A[t%2][i-1]	[j]		[k]) \
															+ 	A[t%2][i+1]	[j]		[k]) \
															+	A[t%2][i]	[j-1]	[k]) \
															+ 	A[t%2][i-1]	[j-1]	[k]) \
															+	A[t%2][i+1]	[j-1]	[k]) \
															+ 	A[t%2][i]	[j+1]	[k]) \
															+	A[t%2][i-1]	[j+1]	[k]) \
															+ 	A[t%2][i+1]	[j+1]	[k]) \
															+	((((((((A[t%2][i]	[j]		[k-1] \
															+	A[t%2][i-1]	[j]		[k-1]) \
															+	A[t%2][i+1]	[j]		[k-1]) \
															+	A[t%2][i]	[j-1]	[k-1]) \
															+	A[t%2][i-1]	[j-1]	[k-1]) \
															+	A[t%2][i+1]	[j-1]	[k-1]) \
															+	A[t%2][i]	[j+1]	[k-1]) \
															+	A[t%2][i-1]	[j+1]	[k-1]) \
															+	A[t%2][i+1]	[j+1]	[k-1]) \
															+ 	((((((((A[t%2][i]	[j]		[k+1] \
															+	A[t%2][i-1]	[j]		[k+1]) \
															+	A[t%2][i+1]	[j]		[k+1]) \
															+	A[t%2][i]	[j-1]	[k+1]) \
															+	A[t%2][i-1]	[j-1]	[k+1]) \
															+	A[t%2][i+1]	[j-1]	[k+1]) \
															+	A[t%2][i]	[j+1]	[k+1]) \
															+	A[t%2][i-1]	[j+1]	[k+1]) \
															+	A[t%2][i+1]	[j+1]	[k+1]))


#define 			Compute_1vector(v_center_0, \
                                    v_center_1, \
                                    v_center_2, \
									v_center_3, \
                                    v_all_d_1_0, \
                                    v_all_d_1_1, \
                                    v_all_d_1_2, \
                                    v_all_d_2_0, \
                                    v_all_d_2_1, \
                                    v_all_d_2_2) 	v_center_0 = _mm256_mul_pd(\
													_mm256_add_pd(\
													_mm256_add_pd(\
													_mm256_add_pd(\
													_mm256_add_pd(\
													_mm256_add_pd(\
													_mm256_add_pd(\
													_mm256_add_pd(\
													_mm256_add_pd(	v_center_0, \
																	v_center_1), \
																	v_center_2), \
																	v_all_d_1_0), \
																	v_all_d_1_1), \
																	v_all_d_1_2), \
																	v_all_d_2_0), \
																	v_all_d_2_1), \
																	v_all_d_2_2), \
																	vc1)


#else

#define C0 0.54
#define C1 0.03
#define C2 0.02
#define C3 0.01

#define SET_COFF  vec vc1, vc0, vc2, vc3 ;\
				 vallset(vc1,C1);\
				 vallset(vc0,C0);\
				 vallset(vc2,C2);\
				 vallset(vc3,C3)

#define Compute_scalar(A,t,i,j,k)  A[(t+1)%2][i][j][k] = C0 * A[(t)%2][i][j][k] + \
                                        C1 * (A[(t)%2][i-1][j][k] + A[(t)%2][i+1][j][k] + A[(t)%2][i][j-1][k] +\
                                                A[(t)%2][i][j+1][k] + A[(t)%2][i][j][k-1] + A[(t)%2][i][j][k+1]) +\
                                        C2 * (A[(t)%2][i-1][j-1][k-1] + A[(t)%2][i-1][j-1][k+1] + A[(t)%2][i-1][j+1][k-1] +\
                                                A[(t)%2][i-1][j+1][k+1] + A[(t)%2][i+1][j-1][k-1] + A[(t)%2][i+1][j-1][k+1] +\
                                                A[(t)%2][i+1][j+1][k-1] + A[(t)%2][i+1][j+1][k+1]) + \
                                        C3 * (A[(t)%2][i-1][j-1][k] + A[(t)%2][i-1][j][k-1] + A[(t)%2][i-1][j][k+1] +\
                                                A[(t)%2][i-1][j+1][k] + A[(t)%2][i][j-1][k-1] + A[(t)%2][i][j-1][k+1] +\
                                                A[(t)%2][i][j+1][k-1] + A[(t)%2][i][j+1][k+1] + A[(t)%2][i+1][j-1][k] +\
                                                A[(t)%2][i+1][j][k-1] + A[(t)%2][i+1][j][k+1] + A[(t)%2][i+1][j+1][k])


#define 			Compute_1vector(v_center_0, \
                                    v_center_1, \
                                    v_center_2, \
									v_center_3, \
                                    v_all_d_1_0, \
                                    v_all_d_1_1, \
                                    v_all_d_1_2, \
                                    v_all_d_2_0, \
                                    v_all_d_2_1, \
                                    v_all_d_2_2) 	v_center_0 = _mm256_fmadd_pd(vc3,\
																	_mm256_add_pd(_mm256_add_pd(v_all_d_2_1,v_all_d_1_0),v_all_d_1_2), \
																	_mm256_fmadd_pd(vc1,\
																					_mm256_add_pd(_mm256_add_pd(v_center_0,v_all_d_1_1),v_center_2), \
																					_mm256_fmadd_pd(vc2,\
																									_mm256_add_pd(v_all_d_2_0, v_all_d_2_2),\
																									_mm256_mul_pd(vc0,v_center_1))))
#endif


void naive_vec(double * A, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb);
void vectime(double * B, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb);
int checkresult( int NX, int NY, int NZ, double (* A_correct)[ NY+2*YSTART][NZ+2*ZSTART], double (* A)[ NY+2*YSTART][NZ+2*ZSTART] );
void print256_vec(__m256d var, char str[] );
