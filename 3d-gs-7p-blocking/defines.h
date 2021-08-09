#include "../common.h"

void vectime(double *B, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb);

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

//#define scalar_ratio

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
											_mm256_add_pd(vz_plus_1,vy_minus_1),vx_minus_1),vy_plus_1),vx_plus_1),vz_minus_1),vc1))//;\
											cnt += 4;
#endif



#define load_vcenter_1(x, y, z)  _mm256_loadu_pd(&A[x - STRIDE * 0][y + 3][z + 3])
#define load_vcenter_2(x, y, z)  _mm256_loadu_pd(&A[x - STRIDE * 1][y + 2][z + 2])
#define load_vcenter_3(x, y, z)  _mm256_loadu_pd(&A[x - STRIDE * 2][y + 1][z + 1])
#define load_vcenter_4(x, y, z)  _mm256_loadu_pd(&A[x - STRIDE * 3][y + 0][z + 0])



#define load_vx_minus_1_1(x, y, z) _mm256_loadu_pd(&A[x - 1 - STRIDE * 0][y + 3][z + 3])
#define load_vx_minus_1_2(x, y, z) _mm256_loadu_pd(&A[x - 1 - STRIDE * 1][y + 2][z + 2])
#define load_vx_minus_1_3(x, y, z) _mm256_loadu_pd(&A[x - 1 - STRIDE * 2][y + 1][z + 1])
#define load_vx_minus_1_4(x, y, z) _mm256_loadu_pd(&A[x - 1 - STRIDE * 3][y + 0][z + 0])


#define load_vx_plus_1_1(x, y, z) _mm256_loadu_pd(&A[x - 1 - STRIDE * 0][y + 2][z + 3])
#define load_vx_plus_1_2(x, y, z) _mm256_loadu_pd(&A[x - 1 - STRIDE * 1][y + 1][z + 2])
#define load_vx_plus_1_3(x, y, z) _mm256_loadu_pd(&A[x - 1 - STRIDE * 2][y + 0][z + 1])
#define load_vx_plus_1_4(x, y, z) _mm256_loadu_pd(&A[x - 1 - STRIDE * 3][y - 1][z + 0])



#define load_real_vx_plus_1_1(x, y, z) _mm256_loadu_pd(&A[x + 1 - STRIDE * 0][y + 3][z + 3])
#define load_real_vx_plus_1_2(x, y, z) _mm256_loadu_pd(&A[x + 1 - STRIDE * 1][y + 2][z + 2])
#define load_real_vx_plus_1_3(x, y, z) _mm256_loadu_pd(&A[x + 1 - STRIDE * 2][y + 1][z + 1])
#define load_real_vx_plus_1_4(x, y, z) _mm256_loadu_pd(&A[x + 1 - STRIDE * 3][y + 0][z + 0])


#define load_vy_plus_1_1(x, y, z) _mm256_loadu_pd(&A[x - STRIDE * 0][y + 1 + 3][z + 3])
#define load_vy_plus_1_2(x, y, z) _mm256_loadu_pd(&A[x - STRIDE * 1][y + 1 + 2][z + 2])
#define load_vy_plus_1_3(x, y, z) _mm256_loadu_pd(&A[x - STRIDE * 2][y + 1 + 1][z + 1])
#define load_vy_plus_1_4(x, y, z) _mm256_loadu_pd(&A[x - STRIDE * 3][y + 1 + 0][z + 0])

#define load_vy_minus_1_1(x, y, z) _mm256_loadu_pd(&A[x - STRIDE * 0][y - 1 + 3][z + 3])
#define load_vy_minus_1_2(x, y, z) _mm256_loadu_pd(&A[x - STRIDE * 1][y - 1 + 2][z + 2])
#define load_vy_minus_1_3(x, y, z) _mm256_loadu_pd(&A[x - STRIDE * 2][y - 1 + 1][z + 1])
#define load_vy_minus_1_4(x, y, z) _mm256_loadu_pd(&A[x - STRIDE * 3][y - 1 + 0][z + 0])



#define store_newvalue_1(vz_minus_1, x, y, z) _mm256_storeu_pd(&A[x - STRIDE * 0][y + 3][z + 3], vz_minus_1)
#define store_newvalue_2(vz_minus_1, x, y, z) _mm256_storeu_pd(&A[x - STRIDE * 1][y + 2][z + 2], vz_minus_1)
#define store_newvalue_3(vz_minus_1, x, y, z) _mm256_storeu_pd(&A[x - STRIDE * 2][y + 1][z + 1], vz_minus_1)
#define store_newvalue_4(vz_minus_1, x, y, z) _mm256_storeu_pd(&A[x - STRIDE * 3][y + 0][z + 0], vz_minus_1)


#define store_vx_plus_1_1(vcenter_shift, x, y, z)  _mm256_storeu_pd(&A[x + 1 - STRIDE * 0][y + 3][z + 3], vcenter_shift)
#define store_vx_plus_1_2(vcenter_shift, x, y, z)  _mm256_storeu_pd(&A[x + 1 - STRIDE * 1][y + 2][z + 2], vcenter_shift)
#define store_vx_plus_1_3(vcenter_shift, x, y, z)  _mm256_storeu_pd(&A[x + 1 - STRIDE * 2][y + 1][z + 1], vcenter_shift)
#define store_vx_plus_1_4(vcenter_shift, x, y, z)  _mm256_storeu_pd(&A[x + 1 - STRIDE * 3][y + 0][z + 0], vcenter_shift)

void blocking_parallel_rectangle_scalar(double * B, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb);
void blocking_parallel_rectangle_vector(double * B, int NX, int NY, int NZ, int T, int xb, int yb, int zb, int tb);
void blocking_sequential_parallelogram_scalar(double * B, int NX, int NY, int T, int xb, int yb, int tb);
void blocking_sequential_rectangle_scalar(double * B, int NX, int NY, int T, int xb, int yb, int tb);
void blocking_sequential_parallelable_rectangle_scalar(double * B, int NX, int NY, int T, int xb, int yb, int tb);
int checkresult( int NX, int NY, int NZ, double (* A_correct)[ NY+2*YSTART][NZ+2*ZSTART], double (* A)[ NY+2*YSTART][NZ+2*ZSTART] );

