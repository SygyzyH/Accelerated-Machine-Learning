/* date = May 27th 2022 21:50 PM */

#ifndef ML_H
#define ML_H

#include <cstddef>
#include <cstdlib>
#include <stdarg.h>
#include <stdlib.h>

typedef struct {
    int width;
    int height;
    int stride;

    double *data;
} Matrix2;

typedef struct {
    int ndims;
    int *dimsz;
    int *stride;

    double *data;
} MatrixN;

static inline MatrixN* makeMatrixN(int ndims, ...) {
    MatrixN *v = (MatrixN *) malloc(sizeof(MatrixN))    ;
    
    v->ndims = ndims;
    v->dimsz = (int *) malloc(sizeof(int) * ndims);
    v->stride = (int *) malloc(sizeof(int) * ndims);

    va_list valist;
    va_start(valist, ndims);
    for (int i = 0; i < ndims; i++) {
        v->dimsz[i] = va_arg(valist, int);
        v->stride[i] = va_arg(valist, int);
    } va_end(valist);

    return v;
}

static inline void freeMatrix2(Matrix2 *m2) {
    free(m2->data);
    m2->data = NULL;
    free(m2);
}

static inline void freeMatrixN(MatrixN *mn) {
    free(mn->data);
    mn->data = NULL;
    free(mn->dimsz)
    mn->dimsz = NULL;
    free(mn->stride);
    mn->stride = NULL;
    free(mn);
}

#endif
