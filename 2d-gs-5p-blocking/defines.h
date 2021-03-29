#include "../common.h"


#define STRIDE 2
#define XSTART 1
#define YSTART 4
#define YSLOPE 1
#define XSLOPE 1


#define INIT 1.0 * (rand() % 1024)
//#define scalarratio

#ifdef simplestencil

#define C0 0.200000
#define SET_COFF 		__m256d vc0 = _mm256_set1_pd(C0)
// note that to boost the performance, values in previous time should be added prior to the ones in the current time
#define Compute_scalar(A,i,j) A[i][j] =  (((((((( A[i][j] )) \
                        				+ A[i][j+1]) + A[i+1][j]) + A[i-1][j]) \
										) + A[i][j-1]) ) * C0
#define Compute_1vector( v3, v1, v4, v7, v5 )			v4=_mm256_mul_pd(\
																	_mm256_add_pd(\
																	_mm256_add_pd(\
																	_mm256_add_pd(\
																	_mm256_add_pd(v4,v7),v5),v3),v1),vc0)

#else

#define C0 0.5
#define C1 0.125
#define SET_COFF 		__m256d vc0 = _mm256_set1_pd(C0);\
						__m256d vc1 = _mm256_set1_pd(C1)
// note that to boost the performance, values in previous time should be added prior to the ones in the current time
#define Compute_scalar(A,i,j) A[i][j] =  C0 * A[i][j] +  (A[i][j+1] + A[i+1][j] + A[i-1][j] + A[i][j-1])  * C1
#define Compute_1vector( v3, v1, v4, v7, v5 )			v4=_mm256_add_pd(_mm256_mul_pd(v4, vc0),\
																	_mm256_mul_pd(\
																	_mm256_add_pd(\
																	_mm256_add_pd(\
																	_mm256_add_pd(v7,v5),v3),v1),vc1))//;\
														cnt+=4;
#endif

void naive(double * B, int NX, int NY, int T, int xb, int yb, int tb);
void vectime(double * B, int NX, int NY, int T, int xb, int yb, int tb);
int checkresult( int NX, int NY, double (* A)[ NY+2*YSTART], double (* B)[ NY+2*YSTART] );

