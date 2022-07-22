#ifndef MAT_H
#define MAT_H

#include "../acceleration/oclapi.h"
#include <stdio.h>
#include <string.h>

#define MAT_PRINT_ERR
// TODO: Is this a good idea?
#ifdef MAT_PRINT_ERR
#ifdef printf_m 
#undef printf_m
#endif
#include <strings.h>
#define printf_m(str, ...) fprintf(stderr, "MAT_H: %s: Error: " str, __func__, __VA_ARGS__)
#else
#define printf_m(str, ...)
#endif

// Tensor accelerated
// Guarenteed to be contigues
// ndims = 0 => scalar.
typedef struct {
    unsigned *dimsz;
    unsigned ndims;

    size_t literal_size;

    double *data;
} Tensor;

typedef enum {
    MAT_NO_ERROR=0,
    MAT_INITIALIZATION_FAILED,
    MAT_UNINITIALIZED,
    MAT_DIMENSION_MISTMATCH,
    MAT_DIMENSION_OUT_OF_RANGE,
    MAT_DIMENSION_ZERO,
    MAT_KERNEL_FAILURE,
    MAT_UNFIT_TENSORS,
    MAT_TENSOR_NO_DATA,
    MAT_TENSOR_NO_DIMS,
    MAT_NULL_PTR
} MatrixErr;

MatrixErr matInit();
Tensor* matMakeTensor(unsigned ndims, unsigned *dims, MatrixErr *e);
Tensor* matTensorDeepCopy(Tensor *t, MatrixErr *e);
double* matTensorAtI(Tensor *t, unsigned *ind, MatrixErr *e);
unsigned *matTensorIAt(Tensor *t, int literal, MatrixErr *e);
MatrixErr matTensorFit(Tensor *t1, Tensor *t2, Tensor **t1r, Tensor **t2r);
void matTensorPrint(Tensor *t);

MatrixErr matProd(Tensor *t1, Tensor *t2, Tensor **r);
MatrixErr matMult(Tensor *t1, Tensor *t2, Tensor **r);
MatrixErr matDot(Tensor *t1, Tensor *t2, Tensor **r);
MatrixErr matAdd(Tensor *t1, Tensor *t2, Tensor **r);
MatrixErr matSub(Tensor *t1, Tensor *t2, Tensor **r);

MatrixErr matTTensor(Tensor *t, Tensor **r);

MatrixErr matSum(double *src, int size, double *res);

static const char* matGetErrorString(MatrixErr error) {
    switch (error) {
        case MAT_NO_ERROR: return "MAT_NO_ERROR";
        case MAT_INITIALIZATION_FAILED: return "MAT_INITIALIZATION_FAILED";
        case MAT_UNINITIALIZED: return "MAT_UNINITIALIZED";
        case MAT_DIMENSION_MISTMATCH: return "MAT_DIMENSION_MISTMATCH";
        case MAT_DIMENSION_OUT_OF_RANGE: return "MAT_DIMENSION_OUT_OF_RANGE";
        case MAT_DIMENSION_ZERO: return "MAT_DIMENSION_ZERO";
        case MAT_KERNEL_FAILURE: return "MAT_KERNEL_FAILURE";
        case MAT_UNFIT_TENSORS: return "MAT_UNFIT_TENSORS";
        case MAT_TENSOR_NO_DATA: return "MAT_TENSOR_NO_DATA";
        case MAT_TENSOR_NO_DIMS: return "MAT_TENSOR_NO_DIMS";
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

static void matFreeTensor(Tensor **t) {
    if (t == NULL) return;

    Tensor *t_d = *t;
    if (t_d != NULL) {
        free(t_d->data);
        t_d->data = NULL;
        free(t_d->dimsz);
        t_d->dimsz = NULL;
    }

    free(*t);
    *t = NULL;
}

static void matFreeTensorD(Tensor t) {
    free(t.data);
    t.data = NULL;
    free(t.dimsz);
    t.dimsz = NULL;
}

static MatrixErr matCheckTensor(Tensor *t, MatrixErr *e) {
    MatrixErr error = MAT_NO_ERROR;

    if (t == NULL) error = MAT_NULL_PTR;
    else if (t->data == NULL) error = MAT_TENSOR_NO_DATA;
    else if (t->dimsz == NULL) error = MAT_TENSOR_NO_DIMS;

    if (e != NULL) *e = error;

    return error;
}

static inline int matIsTensorScalar(Tensor *t) {
    if (t == NULL) return 0;
    return t->literal_size == 1;
}

static inline Tensor* matMakeScalar(double s, MatrixErr *e) {
    Tensor *t = matMakeTensor(0, NULL, e);
    // NOTE: Redundent if.
    if (t != NULL) {
        t->dimsz = (unsigned *) malloc(sizeof(unsigned));
        t->data = (double *) malloc(sizeof(double));
        t->data[0] = s;
        t->dimsz[0] = 1;
    }
    
    return t;
}

static inline Tensor* matTensorFlatten(Tensor *t, MatrixErr *e) {
    if (matCheckTensor(t, e) != MAT_NO_ERROR) return NULL;
 
    unsigned *dimsz = (unsigned *) malloc(sizeof(unsigned));
    dimsz[0] = t->literal_size;
    Tensor *r = matMakeTensor(1, dimsz, NULL);
    free(dimsz);
    r->data = (double *) malloc(sizeof(double) * r->literal_size);
    memcpy((void *) r->data, (void *) t->data, t->literal_size * sizeof(double));

    return r;
}

#endif
