/* date = May 27th 2022 21:50 PM */

#ifndef MAT_H
#define MAT_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "../acceleration/oclapi.h"

// TODO: If `MatrixN.literal_size > MAT_PRINT_CUTOFF`, only print a small part of it.
#ifndef MAT_PRINT_CUTOFF
#define MAT_PRINT_CUTOFF 150
#endif

// Two dimensional matrix struct
typedef struct {
    int width;
    int height;

    double *data;
} Matrix2;

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
} Tensor;

typedef enum {
    MAT_NO_ERROR=0,
    MAT_INITIALIZATION_FAILED,
    MAT_UNINITIALIZED,
    MAT_DIMENSION_MISTMATCH,
    MAT_DIMENSION_OUT_OF_RANGE,
    MAT_KERNEL_FAILURE,
    MAT_UNFIT_TENSORS,
    MAT_NULL_PTR
} MatrixErr;

MatrixErr matInit();
Matrix2* matMakeMatrix2(int width, int height);
Tensor* matMakeTensor(int ndims, int *dims);
Tensor* matMakeTScalar(double s);
double* matNAtI(Tensor m, int *indecies);
int* matNIAt(Tensor m, int literal);
Matrix2* matTensorSubMatrix2(Tensor m, int *start, int width, int height);
double* matTensorContiguousCopy(Tensor m);
MatrixErr matTensorExtendDim(Tensor t, int dim, int rep, Tensor **r);
MatrixErr matSub(Matrix2 m1, Matrix2 m2, Matrix2 **r);
MatrixErr matAdd(Matrix2 m1, Matrix2 m2, Matrix2 **r);
MatrixErr matMul(Matrix2 m1, Matrix2 m2, Matrix2 **r);
MatrixErr matDot(Tensor t1, Tensor t2, Tensor **r);
MatrixErr matSubT(Tensor t1, Tensor t2, Tensor **r);
Matrix2* matT2(Matrix2 m);
Tensor* matTTensor(Tensor m);

static const char* matGetErrorString(MatrixErr error) {
    switch (error) {
        case MAT_NO_ERROR: return "MAT_NO_ERROR";
        case MAT_INITIALIZATION_FAILED: return "MAT_INITIALIZATION_FAILED";
        case MAT_UNINITIALIZED: return "MAT_UNINITIALIZED";
        case MAT_DIMENSION_MISTMATCH: return "MAT_DIMENSION_MISTMATCH";
        case MAT_DIMENSION_OUT_OF_RANGE: return "MAT_DIMENSION_OUT_OF_RANGE";
        case MAT_KERNEL_FAILURE: return "MAT_KERNEL_FAILURE";
        case MAT_UNFIT_TENSORS: return "MAT_UNFIT_TENSORS";
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

static inline void freeMatrix2(Matrix2 *m2) {
    if (m2 != NULL) {
        free(m2->data);
        m2->data = NULL;
    }
    free(m2);
}

static inline void freeTensor(Tensor *t) {
    if (t != NULL) {
        free(t->data);
        t->data = NULL;
        free(t->dimsz);
        t->dimsz = NULL;
        free(t->stride);
        t->stride = NULL;
    }
    free(t);
}

// NOTE: Shouldn't this be the standard?
// Destructor frees the internal logic, user is responsible for freeing the containing structure.
static inline void freeTensorD(Tensor t) {
    free(t.data);
    t.data = NULL;
    free(t.dimsz);
    t.dimsz = NULL;
    free(t.stride);
    t.stride = NULL;
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
static inline int matTPtrAsIndex(Tensor m, double *ptr) {
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
static inline Matrix2* matTAsMatrix2(Tensor m, int width, int height) {
    // Make index an array sized `m.ndims` filled with zeros.
    int *start = (int *) malloc(sizeof(int) * m.ndims);
    for (int i = 0; i < m.ndims; i++) start[i] = 0;

    // Get the submatrix where the start is { 0, 0, 0... } repeated `m.ndims`.
    return matTensorSubMatrix2(m, start, width, height);
}

// Make a `MatrixN` instance out of a Matrix2, containing the same values.
/*
 * Makes a deep copy, with the same dimensions.
 * `m` - input matrix.
 * returns the new matrix. 
 * */
static inline Tensor* mat2AsTensor(Matrix2 m) {
    // New matrix is just a matrix with two dimesions.
    Tensor *r = matMakeTensor(2, (int []) { m.width, m.height });
    r->data = (double *) malloc(sizeof(double) * r->literal_size);

    // Deep copy data.
    if (r != NULL) for (int i = 0; i < r->literal_size; i++) r->data[i] = m.data[i];
    
    return r;
}

// Print `Matrix2`.
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

static void matPrintTensor(Tensor m) {
    int *ind = calloc(m.ndims, sizeof(int));
    // For any even dimension, print linearly. Including 0.
    // For any odd dimension, print newline.
    
    int even_iter = 1;
    for (int i = 0; i < m.ndims; i += 2) even_iter *= m.dimsz[i];
    int odd_iter = 1;
    for (int i = 1; i < m.ndims; i += 2) odd_iter *= m.dimsz[i];

    for (int odd = 0; odd < odd_iter; odd++) {
        // Accounting for the initial `│`
        int printed_even = -2;

        for (int i = 0; i < m.ndims; i += 2) {
            // This has to be done manually, since the `│` character is extended ASCII.
            printed_even += 2;
            printf("│ ");
        }

        for (int even = 0; even < even_iter; even++) {
            printed_even += printf("%#.6g%s", *matNAtI(m, ind), (ind[0] == m.dimsz[0] - 1)? "" : ", ");
            
            // Increment and carry index.
            ind[0]++;
            for (int i = 0; i < m.ndims; i += 2) {
                if (ind[i] >= m.dimsz[i]) {
                    printed_even += 2;
                    printf("│ ");
                    ind[i] = 0;
                    if (i + 2 < m.ndims) ind[i + 2]++;
                }
            }
        } puts("");

        if (m.ndims > 1) ind[1]++;
        for (int i = 1; i < m.ndims; i+= 2) {
            if (ind[i] >= m.dimsz[i]) {
                printf("├");
                for (int j = 0; j < printed_even - 1; j++) printf("─");
                puts("┤");
                ind[i] = 0;
                if (i + 2 < m.ndims) ind[i + 2]++;
            }
        }
    }

    for (int i = 0; i < m.ndims - 1; i++) printf("%dx", m.dimsz[i]);
    printf("%d\n", m.dimsz[m.ndims - 1]);
}

static inline int matTensorIsScalar(Tensor t) {
    for (int i = 0; i < t.ndims; i++) if (t.dimsz[i] != 1) return 0;
    return 1;
}

static inline Tensor* matTensorReduce(Tensor t) {
    int reduce_dims = 0;
    for (int i = 0; i < t.ndims; i++) if (t.dimsz[i] == 1) reduce_dims++;
    int new_ndims = t.ndims - reduce_dims;
    // Reducing scalar
    if (new_ndims < 1) new_ndims = 1;

    int *new_dims = (int *) malloc(sizeof(int) * new_ndims);
    int i_dim = 0;
    for (int i = 0; i < t.ndims; i++) if (t.dimsz[i] != 1) new_dims[i_dim++] = t.dimsz[i];
    // Reducing scalar
    if (t.ndims - reduce_dims < 1) new_dims[0] = 1;

    Tensor *r = matMakeTensor(new_ndims, new_dims);
    r->data = matTensorContiguousCopy(t);

    return r;
}

static MatrixErr matTensorFit(Tensor t1, Tensor t2, Tensor **t1r, Tensor **t2r) {
    if (t1r == NULL) return MAT_NULL_PTR;
    *t1r = NULL;
    if (t2r == NULL) return MAT_NULL_PTR;
    *t2r = NULL;

    // MatrixErr matTensorExtendDim(Tensor t, int dim, int rep, Tensor **r);
    // First, reduce dimensions
    Tensor *new_t1 = matTensorReduce(t1);
    Tensor *new_t2 = matTensorReduce(t2);
    t1 = *new_t1;
    t2 = *new_t2;

    // Biggest dimension out of the two is the target dimension
    Tensor biggest = (t1.ndims > t2.ndims)? t1 : t2;
    int *t1_ones_appended = (int *) malloc(sizeof(int) * biggest.ndims);
    int *t2_ones_appended = (int *) malloc(sizeof(int) * biggest.ndims);

    for (int i = 0; i < biggest.ndims; i++) {
        if (i < t1.ndims) t1_ones_appended[i] = t1.dimsz[i];
        else t1_ones_appended[i] = 1;
    }
    for (int i = 0; i < biggest.ndims; i++) {
        if (i < t2.ndims) t2_ones_appended[i] = t2.dimsz[i];
        else t2_ones_appended[i] = 1;
    }

    for (int i = 0; i < biggest.ndims; i++) {
        if (t1_ones_appended[i] != t2_ones_appended[i] && \
            (t1_ones_appended[i] != 1 && t2_ones_appended[i] != 1)) {
            freeTensor(new_t1);
            freeTensor(new_t2);
            free(t1_ones_appended);
            free(t2_ones_appended);

            return MAT_UNFIT_TENSORS;
        }
    }

    // Since only ones are appended, dimensions are exactly the same
    Tensor *t1_res = matMakeTensor(biggest.ndims, t1_ones_appended);
    t1_res->data = matTensorContiguousCopy(t1);
    Tensor *t2_res = matMakeTensor(biggest.ndims, t2_ones_appended);
    t2_res->data = matTensorContiguousCopy(t2);

    free(t1_ones_appended);
    free(t2_ones_appended);
    
    // Expand dimensions
    for (int i = 0; i < t1_res->ndims; i++) {
        if (t1_res->dimsz[i] == 1 && t2_res->dimsz[i] != 1) {
            Tensor *new_t1_res;
            matTensorExtendDim(*t1_res, i, t2_res->dimsz[i], &new_t1_res);
            freeTensor(t1_res);
            t1_res = new_t1_res;
        }
    }
    for (int i = 0; i < t2_res->ndims; i++) {
        if (t2_res->dimsz[i] == 1 && t1_res->dimsz[i] != 1) {
            Tensor *new_t2_res;
            matTensorExtendDim(*t2_res, i, t1_res->dimsz[i], &new_t2_res);
            freeTensor(t2_res);
            t2_res = new_t2_res;
        }
    }
    
    *t1r = t1_res;
    *t2r = t2_res;

    return MAT_NO_ERROR;
}

#endif
