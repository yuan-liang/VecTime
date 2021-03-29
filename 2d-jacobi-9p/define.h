#include "../common.h"

#define STRIDE 2

#define XSTART 1
#define YSTART 4
#define YSLOPE 1
#define XSLOPE 1

#define INIT 1.0 * (rand() % 1024)


//#define simplestencil
#ifdef simplestencil


#define C0 0.111111111111


#define SET_COFF  __m256d vc0 = _mm256_set1_pd(C0) 


#define Compute_1vector(v_x_minus_2, v_center_2, v_x_plus_2,\
                        v_x_minus_1, v_center_1, v_x_plus_1,\
                        v_x_minus_0, v_center_0, v_x_plus_0)  v_x_minus_0 =  _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(\
						v_x_minus_0, v_center_0), v_x_plus_0),\
                        v_x_minus_1), v_center_1), v_x_plus_1),\
                        v_x_minus_2), v_center_2), v_x_plus_2),vc0)

#define Compute_scalar(A, t, x, y) A[(t+1)%2][x][y] = C0 * \
											(A[t%2][x-1][y-1] + A[t%2][x][y-1] + A[t%2][x+1][y-1] + \
											A[t%2][x-1][y] + A[t%2][x][y] + A[t%2][x+1][y] + \
											 A[t%2][x-1][y+1] + A[t%2][x][y+1] + A[t%2][x+1][y+1]);


#else

#define C0 0.12
#define C1 0.11

#define SET_COFF  __m256d vc0 = _mm256_set1_pd(C0); __m256d vc1 = _mm256_set1_pd(C1)


#define Compute_1vector(v_x_minus_2, v_center_2, v_x_plus_2,\
                        v_x_minus_1, v_center_1, v_x_plus_1,\
                        v_x_minus_0, v_center_0, v_x_plus_0)  v_x_minus_0 =  \
                       _mm256_add_pd( _mm256_mul_pd(v_center_1, vc0), _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(\
						v_x_minus_0, v_center_0), v_x_plus_0),\
                        v_x_minus_1), v_x_plus_1),\
                        v_x_minus_2), v_center_2), v_x_plus_2),vc1))
                        
#define Compute_scalar(A, t, x, y) A[(t+1)%2][x][y] = C0 * A[t%2][x][y] + C1 * \
											(A[t%2][x-1][y-1] + A[t%2][x][y-1] + A[t%2][x+1][y-1] + \
											 A[t%2][x-1][y]                    + A[t%2][x+1][y] + \
											 A[t%2][x-1][y+1] + A[t%2][x][y+1] + A[t%2][x+1][y+1]);

#endif
                            
void vectime(double* A, int NX, int NY, int T);
void naive_scalar(double * A, int NX, int NY, int T);
void naive_vector(double * A, int NX, int NY, int T);
int checkresult( int NX, int NY, double (* A_correct)[ NY+2*YSTART], double (* A)[ NY+2*YSTART]);



