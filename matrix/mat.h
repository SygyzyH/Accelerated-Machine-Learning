#ifndef MAT_H
#define MAT_H

#include "../acceleration/oclapi.h"
#include <string.h>

// TODO: Is this a good idea?
#ifdef MAT_PRINT_ERR
#ifdef printf_m 
#undef printf_m
#endif
#define printf_m(str, ...) fprintf(stderr, "MAT_H " __func__ "Error: " str, __VA_ARGS__)
#else
#define printf_m(str, ...)
#endif

// Tensor accelerated
// Guarenteed to be contigues
// ndims = 0 => scalar.
typedef struct {
    // Number of dimensions.
    unsigned ndimso;
    // Original dimsz. Used to keep track of changes when tensor is extended.
    unsigned *odimsz;
    // Current dimsz.
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

static Tensor* matMakeTensor(unsigned ndims, unsigned *dims, MatrixErr *e) {
    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    
    t->ndims = ndims;
    t->ndimso = ndims;
    t->odimsz = NULL;

    unsigned *dimsz = ndims? (unsigned *) malloc(sizeof(unsigned) * ndims) : NULL;
    int literal_size = 0;
    for (int i = 0; i < ndims; i++) {
        if (dims[i] == 0) {
            free(t);
            free(dimsz);

            if (e != NULL) *e = MAT_DIMENSION_ZERO;
            
            return NULL;
        }

        dimsz[i] = dims[i];
        literal_size *= dims[i];
    }

    t->dimsz = dimsz;
    t->literal_size = literal_size;
    t->data = NULL;

    if (e != NULL) *e = MAT_NO_ERROR;

    return t;
}

static void matFreeTensor(Tensor **t) {
    if (t == NULL) return;

    Tensor *t_d = *t;
    if (t_d != NULL) {
        free(t_d->data);
        free(t_d->dimsz);
        free(t_d->odimsz);
    }

    free(*t);
    *t = NULL;
}

static MatrixErr matCheckTensor(Tensor *t, MatrixErr *e) {
    MatrixErr error = MAT_NO_ERROR;

    if (t == NULL) error = MAT_NULL_PTR;
    else if (t->data == NULL) error = MAT_TENSOR_NO_DATA;
    else if (t->dimsz == NULL) error = MAT_TENSOR_NO_DIMS;

    if (e != NULL) *e = error;

    return error;
}

static Tensor* matMakeScalar(double s, MatrixErr *e) {
    Tensor *t = matMakeTensor(0, NULL, e);
    // NOTE: Redundent if.
    if (t != NULL) t->data[0] = s;
    
    return t;
}

static Tensor* matTensorDeepCopy(Tensor *t, MatrixErr *e) {
    if (t == NULL) {
        if (e != NULL) *e = MAT_NULL_PTR;
        
        return NULL;
    }
    Tensor *r = matMakeTensor(t->ndims, t->dimsz, e);

    if (r == NULL) return NULL;

    if (t->data != NULL) memcpy((void *) r->data, (void *) t->data, t->literal_size);

    if (e != NULL) *e = MAT_NO_ERROR;

    return t;
}

static MatrixErr matTensorFit(Tensor *t1, Tensor *t2, Tensor **t1r, Tensor **t2r) {
    if (t1r == NULL || t2r == NULL) return MAT_NULL_PTR;
    *t1r = NULL;
    *t2r = NULL;
    {
        MatrixErr err;
        if (matCheckTensor(t1, &err) != MAT_NO_ERROR) return err;
        if (matCheckTensor(t2, &err) != MAT_NO_ERROR) return err;
    }
    
    Tensor *biggest = (t1->ndims > t2->ndims)? t1 : t2;
    unsigned *t1_dims = (unsigned *) malloc(sizeof(unsigned) * biggest->ndims);
    unsigned *t2_dims = (unsigned *) malloc(sizeof(unsigned) * biggest->ndims);
    
    for (int i = 0; i < t1->ndims; i++) {
        if (t1->dimsz[i] != 1) 
            t1_dims[i] = t1->dimsz[i];
        else if (i < t2->ndims)
            t1_dims[i] = t2->dimsz[i];
        else
            t1_dims[i] = 1;
    }
    for (int i = 0; i < t2->ndims; i++) {
        if (t2->dimsz[i] != 1) 
            t2_dims[i] = t2->dimsz[i];
        else if (i < t1->ndims)
            t2_dims[i] = t1->dimsz[i];
        else
            t2_dims[i] = 1;
    }

    for (int i = 0; i < biggest->ndims; i++) {
        if (t1_dims[i] != t2_dims[i]) {
            free(t1_dims);
            free(t2_dims);

            return MAT_UNFIT_TENSORS;
        }
    }

    Tensor *t1_res = (Tensor *) malloc(sizeof(Tensor));
    t1_res->ndims = biggest->ndims;
    t1_res->literal_size = t1->literal_size;
    t1_res->dimsz = t1_dims;
    t1_res->odimsz = t1->dimsz;
    t1_res->ndimso = t1->ndims;
    t1_res->data = t1->data;
    
    Tensor *t2_res = (Tensor *) malloc(sizeof(Tensor));
    t2_res->ndims = biggest->ndims;
    t2_res->literal_size = t2->literal_size;
    t2_res->dimsz = t2_dims;
    t2_res->odimsz = t2->dimsz;
    t2_res->ndimso = t2->ndims;
    t2_res->data = t2->data;

    *t1r = t1_res;
    *t2r = t2_res;

    return MAT_NO_ERROR;
}

static double* matTensorAtI(Tensor *t, unsigned *ind, MatrixErr *e) {
    {
        MatrixErr err;
        if (matCheckTensor(t, &err) != MAT_NO_ERROR) {
            if (e != NULL) *e = err;
            
            return NULL;
        }
    }

    double *r = t->data;

    for (int i = 0; i < t->ndims; i++) {
        if (t->odimsz != NULL && (i > t->ndimso || t->dimsz[i] > t->odimsz[i]))
            r += sizeof(double) * (t->dimsz[i] % t->odimsz[i]);
        else r += sizeof(double) * t->dimsz[i];
    }

    if (e != NULL) *e = MAT_NO_ERROR;

    return r;
}

#endif
