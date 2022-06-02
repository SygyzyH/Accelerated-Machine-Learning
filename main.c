#include <stdlib.h>
#include <stdio.h>

#include "matrix/mat.h"
#include "acceleration/oclapi.h"
#include <acceleration/kernels/static_kernels_src.h>

int main() {
    int error;

    error = claInit();
    printf("err: %s\n", claGetErrorString(error));
    printf("oclapi internal: %s\n", clGetErrorString(claGetExtendedError()));

    error = matInit();
    
    //const int width = 2, height = 2;
    Matrix2 *m1 = makeMatrix2(3, 2);
    Matrix2 *m2 = makeMatrix2(4, 3);

    m1->data = (double []) { 2, 1, 1, 1, 1, 1 };
    m2->data = (double []) { 3, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1 };
    matPrintMatrix2(*m2);

    Matrix2 *r;
    printf("error: %s\n", matGetErrorString(matMul(*m1, *m2, &r)));

    puts("result: ");
    if (r != NULL)
        matPrintMatrix2(*r);
    else
        printf("NULL");
    
    return 0;
}
