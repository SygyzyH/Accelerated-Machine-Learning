#include "mat.h"
#include "../acceleration/oclapi.h"
#include <acceleration/kernels/static_kernels_src.h>
#include <stdbool.h>

bool matinit = false;

MatrixErr matInit() {
    if (matinit) return MAT_NO_ERROR;

    // Source code defined in "acceleration/kernels/static_kernels_src.h"
    const char *src_kernel = KERNEL_STATIC_SOURCE_MAT_CL;
    
    claRegisterFromSrc(&src_kernel, 5, "matmul", "matadd", "matsub", "matmuls", "matadds");
    if (claGetError()) return MAT_INITIALIZATION_FAILED;
    
    matinit = true;
    
    return MAT_NO_ERROR;
}

// Make Matrix2 with given dimensions.
/* 
 * The data fielf is not populated, intead the user
 * is expected to allocate `height` * `width` memory,
 * or use a pointer that is at least that size.
 *
 * `width` - width of the matrix.
 * `height` - height of the matrix.
 * returns pointer to generated matrix, or NULL.
 * */
Matrix2* makeMatrix2(int width, int height) {
    if (width <= 0 || height <= 0) return NULL;

    Matrix2 *m2 = (Matrix2 *) malloc(sizeof(Matrix2));
    
    m2->width = width;
    m2->height = height;
    m2->data = NULL;

    return m2;
}

// Make MatrixN with given dimensions.
/*
 * The `data` field is not populated,
 * intead the expected length of the data is put in
 * `literal_size`. The user is expected to allocate 
 * this much memory, or use a pointer that is at least
 * `literal_size`.
 * 
 * `ndims` - number of dimensions.
 * `dims` - dimension size.
 * returns pointer to generated matrix, or NULL.
 * */
MatrixN* makeMatrixN(int ndims, int *dims) {
    if (ndims <= 0) return NULL;

    MatrixN *mn = (MatrixN *) malloc(sizeof(MatrixN));
    
    mn->ndims = ndims;
    mn->dimsz = (int *) malloc(sizeof(int) * ndims);
    mn->stride = (int *) malloc(sizeof(int) * ndims);
    mn->offset = 0;
    mn->data = NULL;

    int sum_strides = 1;

    // Compute stride as the product of all previus
    // strides, multiplied by the current dimension.
    // This is only the initial stride value, and may change
    // when taking sub matricies.
    for (int i = 0; i < ndims; i++) {
        mn->dimsz[i] = dims[i];
        mn->stride[i] = sum_strides;
        
        sum_strides *= mn->dimsz[i];
    }
    
    mn->literal_size = sum_strides;

    return mn;
}

// Get a refrance to the value of a matrix at a given index.
/*
 * `m` - input matrix.
 * `indecies` - indecies to target value.
 * returns a refrence to the value at indecies given.
 * */
double* matNAtI(MatrixN m, int *indecies) {
    if (indecies == NULL) return NULL;

    int literal_index = 0;
    
    for (int i = 0; i < m.ndims; i++) {
        // Index out of bounds for given dimension.
        if (indecies[i] >= m.dimsz[i]) return NULL;

        literal_index += indecies[i] * m.stride[i];
    }
    
    return (double *) (m.data + m.offset + literal_index); 
}

// Return`s the indecies of a literal address for a given matrix.
/*
 * `m` - input matrix.
 * `literal` - literal index to data. This does include offset.
 * returns array of indecies describing the location in the matrix, or NULL.
 * */
int* matNIAt(MatrixN m, int literal) {
    // Index out of matrix bounds
    if (literal > m.literal_size) return NULL;

    int *indecies = (int *) malloc(sizeof(int) * m.ndims);
    // Account for offset.
    literal -= m.offset;
    
    // Reverse the calculations of computing the literal
    // index of an index array.
    for (int i = 0; i < m.ndims; i++) {
        indecies[i] = (literal / m.stride[i]) % m.dimsz[i];
        literal -= indecies[i];
    }

    return indecies;
}

// Make a sub-matrix out of an n dimensional matrix.
/*
 * The matrix will be comprised of `width` * `height` elements
 * of the original matrix `m`, starting from offset `start`.
 * `m` - input matrix.
 * `start` - starting offset.
 * `width` - width of resulting matrix.
 * `height` - height of refsulting matrix.
 * returns sub matrix, or NULL if invalid parameters.
 * */
Matrix2* matNSubMatrix2(MatrixN m, int *start, int width, int height) {
    int effective_start = matNPtrAsIndex(m, matNAtI(m, start));
    
    // If starting index is not in bounds, or otherwise invalid
    // by beign NULL, return NULL.
    if (effective_start == -1) return NULL;
    // This check is technicaly redundent since `matNAtI` returns NULL
    // if out of bounds.
    if (effective_start + width * height > m.literal_size) return NULL;

    Matrix2 *r = makeMatrix2(width, height);
    r->data = (double *) malloc(sizeof(double) * width * height);

    for (int i = 0; i < width * height; i++) {
        int *indecies = matNIAt(m, effective_start + i);
        r->data[i] = *matNAtI(m, indecies);
        free(indecies);
    }

    return r;
}

// Get matrix data as contigues array.
/*
 * Since `MatrixN` is not guarenteed to be contigues,
 * meaning any accsess to it's data requires the use
 * of stride or `matNAtI`, getting the data as a contigues
 * array may be useful when manipulating the entire
 * matrix.
 * `m` - input matrix.
 * returns array of matrix data.
 * */
double* matNContiguousCopy(MatrixN m) {
    double *r = (double *) malloc(sizeof(double) * m.literal_size);

    // Make deep copy linearly.
    for (int i = 0; i < m.literal_size; i++) {
        int *indecies = matNIAt(m, i);
        r[i] = *matNAtI(m, indecies);
        free(indecies);
    }
    
    return r;
}

MatrixErr matMul(Matrix2 m1, Matrix2 m2, Matrix2 **r) {
    // Error checking.
    if (r == NULL) return MAT_NULL_PTR;
    *r = NULL;
    if (!matinit) return MAT_UNINITIALIZED;
    if (m1.width != m2.height) return MAT_DIMENSION_MISTMATCH;
    
    // Standard kernel call in OCLAPI.
    *r = makeMatrix2(m2.width, m1.height);
    Matrix2 *res = *r;
    res->data = (double *) malloc(sizeof(double) * res->width * res->height);

    size_t gz[] = { res->width, res->height };
    claRunKernel("matmul", 2, gz, NULL,
                 m1.data, m1.width * m1.height, OCLREAD | OCLCPY,
                 m2.data, m2.width * m2.height, OCLREAD | OCLCPY,
                 m2.width, m1.width, 
                 res->data, res->width * res->height, OCLWRITE | OCLOUT);
    if (claGetError()) return MAT_KERNEL_FAILURE;

    return MAT_NO_ERROR;
}

MatrixErr matAdd(Matrix2 m1, Matrix2 m2, Matrix2 **r) {
    // Error checking.
    if (r == NULL) return MAT_NULL_PTR;
    *r = NULL;
    if (!matinit) return MAT_UNINITIALIZED;
    if (m1.width != m2.width || m1.height != m2.height) return MAT_DIMENSION_MISTMATCH;
    
    // Standard kernel call in OCLAPI.
    *r = makeMatrix2(m1.width, m1.height);
    Matrix2 *res = *r;
    res->data = (double *) malloc(sizeof(double) * res->width * res->height);

    size_t gz[] = { res->width, res->height };
    claRunKernel("matadd", 2, gz, NULL, 
                 m1.data, m1.width * m1.height, OCLREAD | OCLCPY,
                 m2.data, m2.width * m2.height, OCLREAD | OCLCPY,
                 m1.width, m1.height,
                 res->data, res->width * res->height, OCLWRITE | OCLOUT);
    if (claGetError()) return MAT_KERNEL_FAILURE;

    return MAT_NO_ERROR;
}
