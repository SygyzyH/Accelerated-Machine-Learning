/* date = May 27th 2022 21:50 PM */

#ifndef ML_H
#define ML_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "../acceleration/oclapi.h"

// Two dimensional matrix struct
typedef struct {
    int width;
    int height;
    int stride;

    double *data;
} Matrix2;

// N dimensional matrix struct
typedef struct {
    int ndims;
    int *dimsz;
    int *stride;
    int offset;

    double *data;
    int literal_size;
} MatrixN;

typedef enum {
    MAT_NO_ERROR,
    MAT_INITIALIZATION_FAILED,
    MAT_UNINITIALIZED,
    MAT_DIMENSION_MISTMATCH,
    MAT_KERNEL_FAILURE
} MatrixErr;

MatrixErr matInit();
MatrixErr matAdd(Matrix2 m1, Matrix2 m2, Matrix2 **r);
MatrixErr matMul(Matrix2 m1, Matrix2 m2, Matrix2 **r);

static const char *matGetErrorString(MatrixErr error) {
    switch (error) {
        case MAT_NO_ERROR: return "MAT_NO_ERROR";
        case MAT_INITIALIZATION_FAILED: return "MAT_INITIALIZATION_FAILED";
        case MAT_UNINITIALIZED: return "MAT_UNINITIALIZED";
        case MAT_DIMENSION_MISTMATCH: return "MAT_DIMENSION_MISTMATCH";
        case MAT_KERNEL_FAILURE: return "MAT_KERNEL_FAILURE";
    }
}

static inline void matPrintMatrix2(Matrix2 m) {
    for (int i = 0; i < m.height; i++) {
        for (int j = 0; j < m.width; j++) {
            if (j == 0) printf("[ ");
            if (j == m.width - 1) printf("%lf ]", m.data[j + i * m.stride]);
            else printf("%lf, ", m.data[j + i * m.stride]);
        } puts("");
    } printf("%dx%d\n", m.width, m.height);
}

// Make Matrix2 with given dimensions.
/* 
 * The data fielf is not populated, intead the user
 * is expected to allocate height * stride memory,
 * or use a pointer that is at least that size.
 *
 * `width` - width of the matrix.
 * `height` - height of the matrix.
 * returns pointer to generated matrix, or NULL.
 * */
static Matrix2* makeMatrix2(int width, int height) {
    if (width <= 0 || height <= 0) return NULL;

    Matrix2 *m2 = (Matrix2 *) malloc(sizeof(Matrix2));
    
    m2->width = width;
    m2->height = height;
    m2->stride = width;

    return m2;
}

// Make MatrixN with given dimensions.
/*
 * ndims arguments describe the size of each dimension
 * of the matrix. The data field is not populated,
 * intead the expected length of the data is put in
 * literal_size. The user is expected to allocate 
 * this much memory, or use a pointer that is at least
 * literal_size.
 * 
 * `ndims` - number of dimensions.
 * `...` - dimension size.
 * returns pointer to generated matrix, or NULL.
 * */
static MatrixN* makeMatrixN(int ndims, ...) {
    if (ndims <= 0) return NULL;

    MatrixN *mn = (MatrixN *) malloc(sizeof(MatrixN));
    
    mn->ndims = ndims;
    mn->dimsz = (int *) malloc(sizeof(int) * ndims);
    mn->stride = (int *) malloc(sizeof(int) * (ndims - 1));
    mn->offset = 0;

    int sum_strides = 1;

    va_list valist;
    va_start(valist, ndims);
    for (int i = 0; i < ndims; i++) {
        mn->dimsz[i] = va_arg(valist, int);
        
        sum_strides *= mn->dimsz[i];
        
        if (i < ndims - 1) mn->stride[i] = sum_strides;
    } va_end(valist);
    
    mn->literal_size = sum_strides;

    return mn;
}

static inline void freeMatrix2(Matrix2 *m2) {
    free(m2->data);
    m2->data = NULL;
    free(m2);
}

static inline void freeMatrixN(MatrixN *mn) {
    free(mn->data);
    mn->data = NULL;
    free(mn->dimsz);
    mn->dimsz = NULL;
    free(mn->stride);
    mn->stride = NULL;
    free(mn);
}

static inline cl_int matGetExtendedError() {
    return claGetExtendedError();
}

#endif
