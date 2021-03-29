#include "../common.h"

#define STRIDE 8
#define XSTART VECLEN
#define XSLOPE  1
#define TSLOPE STRIDE

#define INIT 1.0 * (rand() % 1024)


#define C0 0.3333
#define SET_COFF 		__m256d vc0 = _mm256_set1_pd(C0)

#define Compute_scalar(A,x) A[x] = (A[x-1] + (A[x] + A[x+1])) * C0
#define Compute_1vector(v0,v1,v2)  v1=_mm256_mul_pd(_mm256_add_pd(v0,_mm256_add_pd(v1,v2)),vc0)

void vectime(double* A, int N, int T, int xb, int tb);
void naive_scalar(double * A, int N, int T, int xb, int tb);
int checkresult( int N, double * A_correct, double * A);

