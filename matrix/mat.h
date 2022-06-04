/* date = May 27th 2022 21:50 PM */

#ifndef MAT_H
#define MAT_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "../acceleration/oclapi.h"

// Two dimensional matrix struct
typedef struct {
    int width;
    int height;

    double *data;
} Matrix2;

// Three dimensional matrix struct
typedef struct {
    int width;
    int height;
    int depth;

    double *data;
} Matrix3;

// N dimensional matrix struct
typedef struct {
    int ndims;
    int *dimsz;

    // NOTE: Stride[0] will always be 1.
    // NOTE: For dimension n, at given index k will be at (offset + data)[k * stride[n]] (Not accumelative stride!)
    int *stride;
    int offset;
    
    // NOTE: Array data is not guarenteed to be contigues.
    double *data;
    int literal_size;
} MatrixN;

typedef enum {
    MAT_NO_ERROR,
    MAT_INITIALIZATION_FAILED,
    MAT_UNINITIALIZED,
    MAT_DIMENSION_MISTMATCH,
    MAT_KERNEL_FAILURE,
    MAT_NULL_PTR
} MatrixErr;

MatrixErr matInit();
Matrix2* makeMatrix2(int width, int height);
MatrixN* makeMatrixN(int ndims, int *dims);
double* matNAtI(MatrixN m, int *indecies);
int* matNIAt(MatrixN m, int literal);
Matrix2* matNSubMatrix2(MatrixN m, int *start, int width, int height);
double* matNContiguousCopy(MatrixN m);
MatrixErr matAdd(Matrix2 m1, Matrix2 m2, Matrix2 **r);
MatrixErr matMul(Matrix2 m1, Matrix2 m2, Matrix2 **r);

static const char* matGetErrorString(MatrixErr error) {
    switch (error) {
        case MAT_NO_ERROR: return "MAT_NO_ERROR";
        case MAT_INITIALIZATION_FAILED: return "MAT_INITIALIZATION_FAILED";
        case MAT_UNINITIALIZED: return "MAT_UNINITIALIZED";
        case MAT_DIMENSION_MISTMATCH: return "MAT_DIMENSION_MISTMATCH";
        case MAT_KERNEL_FAILURE: return "MAT_KERNEL_FAILURE";
        case MAT_NULL_PTR: return "MAT_NULL_PTR";
        default: return "Unknown Matrix error";
    }
}

// Returns extended OpenCL error.
/*
 * Resets the last OpenCL API error and return it.
 * */
static inline cl_int matGetExtendedError() {
    return claGetExtendedError();
}

// Returns ptr to value in matrix as a literal index.
/*
 * Mostly used to translate from pointer space when
 * using `matNAtI`. Behavior when pointer is outside of
 * the matrix .data segment is undefined. Passing a
 * NULL pointer will return index -1.
 * `m` - input matrix.
 * `ptr` - pointer to value in matrix.
 * returns index of pointer in matrix.
 * */
static inline int matNPtrAsIndex(MatrixN m, double *ptr) {
    return (ptr == NULL)? -1 : (int) (ptr - m.data - m.offset);
}

// Makes a `Matrix2` instance out of a MatrixN, containing the same values.
/*
 * Will make a deep copy.
 * `m` - input matrix.
 * `width` - output matrix width.
 * `height` - output matrix height.
 * returns matrix, or NULL if invalid.
 * */
static inline Matrix2* matNAsMatrix2(MatrixN m, int width, int height) {
    // Make index an array sized `m.ndims` filled with zeros.
    int *start = (int *) malloc(sizeof(int) * m.ndims);
    for (int i = 0; i < m.ndims; i++) start[i] = 0;

    // Get the submatrix where the start is { 0, 0, 0... } repeated `m.ndims`.
    return matNSubMatrix2(m, start, width, height);
}

// Make a `MatrixN` instance out of a Matrix2, containing the same values.
/*
 * Makes a deep copy, with the same dimensions.
 * `m` - input matrix.
 * returns the new matrix. 
 * */
static inline MatrixN* mat2AsMatrixN(Matrix2 m) {
    // New matrix is just a matrix with two dimesions.
    MatrixN *r = makeMatrixN(2, (int []) { m.width, m.height });
    r->data = (double *) malloc(sizeof(double) * r->literal_size);

    // Deep copy data.
    if (r != NULL) for (int i = 0; i < r->literal_size; i++) r->data[i] = m.data[i];
    
    return r;
}

// Print matrix.
/*
 * `m` - matrix to print.
 * */
static inline void matPrintMatrix2(Matrix2 m) {
    for (int i = 0; i < m.height; i++) {
        for (int j = 0; j < m.width; j++) {
            if (j == 0) printf("[ ");
            if (j == m.width - 1) printf("%lf ]", m.data[j + i * m.width]);
            else printf("%lf, ", m.data[j + i * m.width]);
        } puts("");
    } printf("%dx%d\n", m.width, m.height);
}

static inline void freeMatrix2(Matrix2 *m2) {
    if (m2 != NULL) {
        free(m2->data);
        m2->data = NULL;
    }
    free(m2);
}

static inline void freeMatrixN(MatrixN *mn) {
    if (mn != NULL) {
        free(mn->data);
        mn->data = NULL;
        free(mn->dimsz);
        mn->dimsz = NULL;
        free(mn->stride);
        mn->stride = NULL;
    }
    free(mn);
}

#endif
