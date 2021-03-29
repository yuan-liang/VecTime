#include "../common.h"

#define VECLEN_INT 8
#define STRIDE 3
#define XSTART 1
#define LCSSTART 1


int checkresult(int N, int * A_correct, int * A);
int naive_scalar(char * x1, char * x2, int nx1, int nx2, int * lcs);
int vectime(char * x1, char * x2, int nx1, int nx2, int * lcs);

