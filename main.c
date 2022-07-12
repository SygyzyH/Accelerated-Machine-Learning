#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "matrix/mat.h"
#include "acceleration/oclapi.h"
#include "ml/ml.h"
#include <acceleration/kernels/static_kernels_src.h>

#include <assert.h>
int main() {
    int error;

    error = claInit();
    printf("err: %s\n", claGetErrorString(error));
    printf("oclapi internal: %s\n", clGetErrorString(claGetExtendedError()));

    error = matInit();

    Tensor *t1 = matMakeTensor(1, (unsigned []) { 2, 3, 3 }, (MatrixErr *) &error);
    t1->data = (double *) malloc(sizeof(double) * t1->literal_size);
    for (int i = 0; i < t1->literal_size; i++) t1->data[i] = i;
    matTensorPrint(t1);

    Tensor *t2 = matMakeTensor(2, (unsigned []) { 2, 2, 3 }, (MatrixErr *) &error);
    t2->data = (double *) malloc(sizeof(double) * t2->literal_size);
    for (int i = 0; i < t2->literal_size; i++) t2->data[i] = i;
    matTensorPrint(t2);

    Tensor *c = matMakeScalar(20, (MatrixErr *) &error);
    //matTensorPrint(c);

    Tensor *r;
    matProd(t1, t2, &r);
    matTensorPrint(r);

    claCln();
    
    return 0;
}
