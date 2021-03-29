#include "../common.h"

#define VECLEN_INT 8
#define STRIDE 3
#define XSTART 1
#define LCSSTART 1



int checkresult(int N, int * A_correct, int * A);
int naive_scalar(char * sx, char * st, int nx, int nt, int bx, int bt, int * lcs);
int vectime(char * sx, char * st, int nx, int nt, int bx, int bt, int * lcs);

