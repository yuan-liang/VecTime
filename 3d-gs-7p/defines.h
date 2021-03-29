#include "../common.h"


#define STRIDE 2

#define XSTART 1
#define YSTART 1
#define ZSTART 4

#define YSLOPE 1
#define XSLOPE 1
#define ZSLOPE 1

#define INIT		1.0 * (rand() % 1024);

#ifdef simplestencil
#define C0			0.143
#define SET_COFF	__m256d vc0 = _mm256_set1_pd(C0)

#define Compute_scalar(A,i,j,k) A[i][j][k] =  ((((((A[i][j][k] + A[i][j][k+1]) + A[i][j-1][k]) + A[i-1][j][k]) + A[i][j+1][k]) + A[i+1][j][k]) + A[i][j][k-1]) * C0

#define Compute_1vector(vcenter,vz_minus_1,vz_plus_1,vx_minus_1, vx_plus_1,vy_minus_1, vy_plus_1)	vcenter = _mm256_mul_pd(\
											_mm256_add_pd(\
											_mm256_add_pd(\
											_mm256_add_pd(\
											_mm256_add_pd(\
											_mm256_add_pd(\
											_mm256_add_pd(vcenter,vz_plus_1),vy_minus_1),vx_minus_1),vy_plus_1),vx_plus_1),vz_minus_1),vc0);vz_minus_1=vcenter
#else
#define C0			0.25
#define C1			0.125
#define SET_COFF	__m256d vc0 = _mm256_set1_pd(C0);\
					__m256d vc1 = _mm256_set1_pd(C1)

#define Compute_scalar(A,i,j,k) A[i][j][k] =  C0 * A[i][j][k] + C1 * (((((( A[i][j][k+1]) + A[i][j-1][k]) + A[i-1][j][k]) + A[i][j+1][k]) + A[i+1][j][k]) + A[i][j][k-1])

#define Compute_1vector(vcenter,vz_minus_1,vz_plus_1,vx_minus_1, vx_plus_1,vy_minus_1, vy_plus_1)	vz_minus_1 = _mm256_add_pd(_mm256_mul_pd(vcenter, vc0),\
											_mm256_mul_pd(\
											_mm256_add_pd(\
											_mm256_add_pd(\
											_mm256_add_pd(\
											_mm256_add_pd(\
											_mm256_add_pd(vz_plus_1,vy_minus_1),vx_minus_1),vy_plus_1),vx_plus_1),vz_minus_1),vc1))
#endif


void vectime(double* A, int NX,int NY, int NZ, int T);	
int checkresult( int NX, int NY, int NZ, double (* A_correct)[ NY+2*YSTART][NZ+2*ZSTART], double (* A)[ NY+2*YSTART][NZ+2*ZSTART]);
