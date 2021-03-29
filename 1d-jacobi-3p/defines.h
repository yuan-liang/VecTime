#include "../common.h"

#define XSTART 4
#define XSLOPE 1
#define STRIDE 7
#define LANESTRIDE 2
#define STRIDE4 4
#define STRIDE11 11

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

#define SET_COFF 		__m256d vc0 = _mm256_set1_pd(C0); __m256d vc1 = _mm256_set1_pd(C1)

#define Compute_4vector(v0,v1,v2,v3,v4,v5) Compute_1vector(v0,v1,v2);\
											Compute_1vector(v1,v2,v3);\
											Compute_1vector(v2,v3,v4);\
											Compute_1vector(v3,v4,v5) 


#define Compute_3vector(v0,v1,v2,v3,v4) Compute_1vector(v0,v1,v2);\
											Compute_1vector(v1,v2,v3);\
											Compute_1vector(v2,v3,v4)

int checkresult(int NX, double * A_correct, double * A) ;
void naive_scalar(double * A, int NX, int T);
void naive_vector(double * A, int NX, int T);
void vectime(double* A, int NX, int T);




#define InOut_POS_0(out, v, in)	out = _mm256_blend_pd(out, v, 0b0001); v = _mm256_blend_pd(v, in, 0b0001)
#define InOut_POS_1(out, v, in)	out = _mm256_blend_pd(out, v, 0b0010); v = _mm256_blend_pd(v, in, 0b0010)
#define InOut_POS_2(out, v, in)	out = _mm256_blend_pd(out, v, 0b0100); v = _mm256_blend_pd(v, in, 0b0100)
#define InOut_POS_3(out, v, in)	out = _mm256_blend_pd(out, v, 0b1000); v = _mm256_blend_pd(v, in, 0b1000)


#define Load_4_Vectors(addr, in0,in1,in2,in3)	in0 = _mm256_loadu_pd(addr);\
													in1 = _mm256_loadu_pd(addr + VECLEN);\
													in2 = _mm256_loadu_pd(addr + 2 * VECLEN);\
													in3 = _mm256_loadu_pd(addr + 3 * VECLEN)

#define Store_4_Vectors(out0,in0,in1,in2,addr)	_mm256_storeu_pd(addr,out0);\
													_mm256_storeu_pd(addr + VECLEN,in0);\
													_mm256_storeu_pd(addr + 2 * VECLEN,in1);\
													_mm256_storeu_pd(addr + 3 * VECLEN,in2)

#define Compute_Inout_Pos_0_Vector(v0,v1,v2,out0,in1)			Compute_1vector(v0,v1,v2);\
													InOut_POS_0(out0, v0, in1)
#define Compute_Inout_Pos_1_Vector(v0,v1,v2,out0,in1)			Compute_1vector(v0,v1,v2);\
													InOut_POS_1(out0, v0, in1)
													
#define Compute_Inout_Pos_2_Vector(v0,v1,v2,out0,in1)			Compute_1vector(v0,v1,v2);\
													InOut_POS_2(out0, v0, in1)
#define Compute_Inout_Pos_3_Vector(v0,v1,v2,out0,in1)			Compute_1vector(v0,v1,v2);\
													InOut_POS_3(out0, v0, in1)									




#define Load_STRIDE_Vectors(addr, in0,in1,in2,in3)	Load_4_Vectors(addr, in0,in1,in2,in3)

#define Store_STRIDE_Vectors(out0,in0,in1,in2,addr)	Store_4_Vectors(out0,in0,in1,in2,addr)

#define Compute_STRIDE_Minus_XSLOPE_Vectors(v0,v1,v2,v3,v4) Compute_3vector(v0,v1,v2,v3,v4)
#define Compute_XSLOPE_vectors(v0,v1,v2) 					Compute_1vector(v0,v1,v2)

#define Shift_left_XSLOPE_vectors(v0,v1)  v1 = _mm256_permute4x64_pd(v0,0b00111001) //	a0 a1 a2 a3 --> a1 a2 a3 a0
#define Shift_right_XSLOPE_vectors(v0,v1) v1 = _mm256_permute4x64_pd(v0,0b10010011) //	a0 a1 a2 a3 --> a3 a0 a1 a2

#define InOut_Pos_0_STRIDE_Minus_XSLOPE_Vectors(v0,v1,v2)	InOut_POS_0(out0, v0, in0);\
															InOut_POS_0(in0, v1, in1);\
															InOut_POS_0(in1, v2, in2)

#define InOut_POS_0_XSLOPE_Vectors(v0)						InOut_POS_0(in2, v0, in3)


#define InOut_Pos_1_STRIDE_Minus_XSLOPE_Vectors(v0,v1,v2)	InOut_POS_1(out0, v0, in0);\
															InOut_POS_1(in0, v1, in1);\
															InOut_POS_1(in1, v2, in2)

#define InOut_POS_1_XSLOPE_Vectors(v0)						InOut_POS_1(in2, v0, in3)




#define InOut_Pos_2_STRIDE_Minus_XSLOPE_Vectors(v0,v1,v2)	InOut_POS_2(out0, v0, in0);\
															InOut_POS_2(in0, v1, in1);\
															InOut_POS_2(in1, v2, in2)

#define InOut_POS_2_XSLOPE_Vectors(v0)						InOut_POS_2(in2, v0, in3)




#define InOut_Pos_3_STRIDE_Minus_XSLOPE_Vectors(v0,v1,v2)	InOut_POS_3(out0, v0, in0);\
															InOut_POS_3(in0, v1, in1);\
															InOut_POS_3(in1, v2, in2)

#define InOut_POS_3_XSLOPE_Vectors(v0)						InOut_POS_3(in2, v0, in3)
