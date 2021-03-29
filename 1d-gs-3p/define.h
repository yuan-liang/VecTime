#include "../common.h"

#define STRIDE 8
#define XSTART 1
#define XSLOPE 1

#define INIT 1.0 * (rand() % 1024)



#define C0 0.33333
#define SET_COFF 		__m256d vc0 = _mm256_set1_pd(C0)

#define Compute_scalar(A,x) A[x] = (A[x-1] + (A[x] + A[x+1])) * C0
#define Compute_1vector(v0,v1,v2)  v1=_mm256_mul_pd(_mm256_add_pd(v0,_mm256_add_pd(v1,v2)),vc0)



void vectime(double* A, int N, int T);
void naive_scalar(double * A, int N, int T);
int checkresult(int N, double * A_correct, double * A);
