#include "../common.h"

#define XSTART 4
#define XSLOPE 1
#define STRIDE 7
#define LANESTRIDE 2
#define STRIDE4 4

#define INIT 1.0 * (rand() % 1024)


#if defined(heat)
#define C0 0.250
#define C1 -2.0
#define Compute_scalar(A, t, x) A[(t+1)%2][x] = C0 * ((A[t%2][x+1] +  C1 * A[t%2][x]) + A[t%2][x-1])
#define Compute_1vector(v0,v1,v2)  v0=_mm256_mul_pd(vc0,_mm256_add_pd(_mm256_fmadd_pd(vc1, v1, v2),v0))

#else

#define C0 0.75
#define C1 0.125
#define Compute_scalar(A, t, x) A[(t+1)%2][x] = C0 * A[t%2][x] +  C1 * (A[t%2][x+1] + A[t%2][x-1])
#define Compute_1vector(v0,v1,v2)  v0=_mm256_fmadd_pd(v1,vc0,_mm256_mul_pd(_mm256_add_pd(v0,v2),vc1))

#endif

#define SET_COFF                __m256d vc0 = _mm256_set1_pd(C0); __m256d vc1 = _mm256_set1_pd(C1)

#define Compute_scalar(A, t, x) A[(t+1)%2][x] = C0 * ((A[t%2][x+1] +  C1 * A[t%2][x]) + A[t%2][x-1])
#define Compute_1vector(v0,v1,v2)  v0=_mm256_mul_pd(vc0,_mm256_add_pd(_mm256_fmadd_pd(vc1, v1, v2),v0))
 
#define Compute_4vector(v0,v1,v2,v3,v4,v5) Compute_1vector(v0,v1,v2);\
											Compute_1vector(v1,v2,v3);\
											Compute_1vector(v2,v3,v4);\
											Compute_1vector(v3,v4,v5) 
											
int checkresult(int NX, double * A_correct, double * A) ;
void naive_vector(double * A, int NX, int T, int xb, int tb);
void vectime(double* A, int N, int T, int Bx, int tb);







