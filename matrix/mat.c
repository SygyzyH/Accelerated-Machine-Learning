#include "mat.h"
#include "../acceleration/oclapi.h"
#include <acceleration/kernels/static_kernels_src.h>
#include <stdbool.h>
#include <stdio.h>

bool matinit = false;

MatrixErr matInit() {
    if (matinit) return MAT_NO_ERROR;

    // Source code defined in "acceleration/kernels/static_kernels_src.h"
    const char *src_kernel = KERNEL_STATIC_SOURCE_MAT_CL;
    
    claRegisterFromSrc(&src_kernel, 6, "matmul", "matadd", "matsub", "matt", "matmuls", "matadds");
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
Matrix2* matMakeMatrix2(int width, int height) {
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
Tensor* matMakeTensor(int ndims, int *dims) {
    if (ndims <= 0) return NULL;
    for (int i = 0; i < ndims; i++) if (dims[i] < 1) return NULL;

    Tensor *mn = (Tensor *) malloc(sizeof(Tensor));
    
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

Tensor* matMakeTScalar(double s) {
    Tensor *r = matMakeTensor(1, (int []) { 1 });
    r->data = (double *) malloc(sizeof(double));
    r->data[0] = s;

    return r;
}

// Get a refrance to the value of a matrix at a given index.
/*
 * `m` - input matrix.
 * `indecies` - indecies to target value.
 * returns a refrence to the value at indecies given.
 * */
double* matNAtI(Tensor m, int *indecies) {
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
int* matNIAt(Tensor m, int literal) {
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
Matrix2* matTensorSubMatrix2(Tensor m, int *start, int width, int height) {
    int effective_start = matTPtrAsIndex(m, matNAtI(m, start));
    
    // If starting index is not in bounds, or otherwise invalid
    // by beign NULL, return NULL.
    if (effective_start == -1) return NULL;
    // This check is technicaly redundent since `matNAtI` returns NULL
    // if out of bounds.
    if (effective_start + width * height > m.literal_size) return NULL;

    Matrix2 *r = matMakeMatrix2(width, height);
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
double* matTensorContiguousCopy(Tensor m) {
    double *r = (double *) malloc(sizeof(double) * m.literal_size);

    // Make deep copy linearly.
    for (int i = 0; i < m.literal_size; i++) {
        int *indecies = matNIAt(m, i);
        r[i] = *matNAtI(m, indecies);
        free(indecies);
    }
    
    return r;
}

MatrixErr matTensorExtendDim(Tensor t, int dim, int rep, Tensor **r) {
    if (r == NULL) return MAT_NULL_PTR;
    *r = NULL;
    if (dim < 0 || dim >= t.ndims) return MAT_DIMENSION_OUT_OF_RANGE;
    
    int *dims = (int *) malloc(sizeof(int) * t.ndims);
    for (int i = 0; i < t.ndims; i++) 
        if (i != dim) dims[i] = t.dimsz[i];
        else dims[i] = t.dimsz[i] * rep;
    Tensor *res = matMakeTensor(t.ndims, dims);
    res->data = (double *) malloc(sizeof(double) * res->literal_size);

    for (int i = 0; i < res->literal_size; i++) {
        int *ind = matNIAt(*res, i);
        if (ind[dim] >= t.dimsz[dim]) {
            double *r_dest = matNAtI(*res, ind);
            ind[dim] %= t.dimsz[dim];
            *r_dest = *matNAtI(t, ind);
        } else *matNAtI(*res, ind) = *matNAtI(t, ind);

        free(ind);
    }

    *r = res;

    return MAT_NO_ERROR;
}

// Standardized mul operator for any two tensors.
// Iterate over the first dimension of the first tensor and the second dimension of the
// second tensor. Sum their product.
// The result is sized t1.dims[1:end]t2.dims[0,2:end]. 
// If the resulting dimension is 0 than a scalar is returned.
MatrixErr matDot(Tensor t1, Tensor t2, Tensor **r) {
    if (r == NULL) return MAT_NULL_PTR;
    *r = NULL;
    Tensor *new_t1 = matTensorReduce(t1);
    Tensor *new_t2 = matTensorReduce(t2);
    t1 = *new_t1;
    t2 = *new_t2;
    int t2_second_dim = (t2.ndims < 2)? 0 : 1;
    if (!matTensorIsScalar(t1) && !matTensorIsScalar(t2)) {
        if (t1.dimsz[0] != t2.dimsz[t2_second_dim]) {
            freeTensor(new_t1);
            freeTensor(new_t2);
            return MAT_DIMENSION_MISTMATCH;
        }
    }

    int res_ndims;
    int *res_dims = NULL;
    if (matTensorIsScalar(t1)) {
        res_ndims = t2.ndims;
        res_dims = t2.dimsz;
    } else if (matTensorIsScalar(t2)) {
        res_ndims = t1.ndims;
        res_dims = t1.dimsz;
    } else {
        res_ndims = t1.ndims + t2.ndims - 2;
        if (res_ndims == 0) {
            res_ndims = 1;
            res_dims = (int *) malloc(sizeof(int) * res_ndims);
            res_dims[0] = 1;
        } else {
            res_dims = (int *) malloc(sizeof(int) * res_ndims);
            for (int i = 0; i < res_ndims; i++) {
                if (i + 1 < t1.ndims)
                    res_dims[i] = t1.dimsz[i + 1];
                else {
                    int t2_corrected_i = i - t1.ndims + 1;
                    if (t2_corrected_i < t2_second_dim)
                        res_dims[i] = t2.dimsz[t2_corrected_i];
                    else res_dims[i] = t2.dimsz[t2_corrected_i + 1];
                }
            }
        }
    }
    Tensor *res = matMakeTensor(res_ndims, res_dims);
    res->data = (double *) malloc(sizeof(double) * res->literal_size);
    if (res_dims != t1.dimsz && res_dims != t2.dimsz) free(res_dims);
    res_dims = NULL;
    
    int *t1_ind = calloc(t1.ndims, sizeof(int));
    int *t2_ind = calloc(t2.ndims, sizeof(int));

    for (int i = 0; i < res->literal_size; i++) {
        double sum = 0;
        for (int com_dimsz = 0; com_dimsz < t1.dimsz[0] && com_dimsz < t2.dimsz[t2_second_dim]; com_dimsz++) {
            double *t1_p = matNAtI(t1, t1_ind);
            double *t2_p = matNAtI(t2, t2_ind);
            sum += *t1_p * *t2_p;

            t1_ind[0]++;
            if (t1_ind[0] > t1.dimsz[0]) t1_ind[0] = 0;
            t2_ind[t2_second_dim]++;
            if (t2_ind[t2_second_dim] > t2.dimsz[t2_second_dim]) t2_ind[t2_second_dim] = 0;
        } 

        int *res_ind = matNIAt(*res, i);
        *matNAtI(*res, res_ind) = sum;
        free(res_ind);
        
        for (int ndims = 0; ndims < t2.ndims; ndims++) {
            if (t2_ind[ndims] >= t2.dimsz[ndims]) {
                t2_ind[ndims] = 0;
                if (t2_second_dim && ndims == t2_second_dim) {
                    t2_ind[ndims - 1]++;
                    if (t2_ind[ndims - 1] >= t2.dimsz[ndims - 1]) {
                        t2_ind[ndims - 1] = 0;
                        t2_ind[ndims + 1]++;
                    }
                } else if (ndims < t2.ndims - 1) t2_ind[ndims + 1]++;
            }
        }
        int final_dim = 1;
        for (int ndims = 0; ndims < t2.ndims; ndims++) if (t2_ind[ndims] != 0) final_dim = 0;
        if (!final_dim) t1_ind[0] = 0;

        for (int ndims = 0; ndims < t1.ndims; ndims++) {
            if (t1_ind[ndims] >= t1.dimsz[ndims]) {
                t1_ind[ndims] = 0;
                if (ndims < t1.ndims - 1) t1_ind[ndims + 1]++;
            }
        }
    }

    free(t1_ind);
    free(t2_ind);

    freeTensor(new_t1);
    freeTensor(new_t2);
    
    *r = res;

    return MAT_NO_ERROR;
}

MatrixErr matSubT(Tensor t1, Tensor t2, Tensor **r) {
    if (r == NULL) return MAT_NULL_PTR;
    *r = NULL;

    Tensor *new_t1;
    Tensor *new_t2;
    MatrixErr error = matTensorFit(t1, t2, &new_t1, &new_t2);

    if (error != MAT_NO_ERROR) {
        freeTensor(new_t1);
        freeTensor(new_t2);

        return error;
    }

    Tensor *res = matMakeTensor(new_t1->ndims, new_t1->dimsz);
    res->data = (double *) malloc(sizeof(double) * res->literal_size);

    for (int i = 0; i < new_t1->literal_size; i++) {
        int *ind1 = matNIAt(*new_t1, i);
        int *ind2 = matNIAt(*new_t2, i);
        int *indr = matNIAt(*res, i);

        *matNAtI(*res, indr) = *matNAtI(*new_t1, ind1) - *matNAtI(*new_t2, ind2);
    }

    *r = res;

    return MAT_NO_ERROR;
}

MatrixErr matMul(Matrix2 m1, Matrix2 m2, Matrix2 **r) {
    // Error checking.
    if (r == NULL) return MAT_NULL_PTR;
    *r = NULL;
    if (!matinit) return MAT_UNINITIALIZED;
    if (m1.width != m2.height) return MAT_DIMENSION_MISTMATCH;
    
    // Standard kernel call in OCLAPI.
    *r = matMakeMatrix2(m2.width, m1.height);
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
    *r = matMakeMatrix2(m1.width, m1.height);
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

MatrixErr matSub(Matrix2 m1, Matrix2 m2, Matrix2 **r) {
    // Error checking.
    if (r == NULL) return MAT_NULL_PTR;
    *r = NULL;
    if (!matinit) return MAT_UNINITIALIZED;
    if (m1.width != m2.width || m1.height != m2.height) return MAT_DIMENSION_MISTMATCH;

    // Standard kernel call in OCLAPI.
    *r = matMakeMatrix2(m1.width, m1.height);
    Matrix2 *res = *r;
    res->data = (double *) malloc(sizeof(double) * res->width * res->height);

    size_t gz[] = { res->width, res->height };
    claRunKernel("matsub", 2, gz, NULL, 
                 m1.data, m1.width * m1.height, OCLREAD | OCLCPY,
                 m2.data, m2.width * m2.height, OCLREAD | OCLCPY,
                 m1.width, m1.height,
                 res->data, res->width * res->height, OCLWRITE | OCLOUT);
    if (claGetError()) return MAT_KERNEL_FAILURE;

    return MAT_NO_ERROR;
}

Matrix2* matT2(Matrix2 m) {
    // Error checking.
    if (!matinit) return NULL;

    // Standard kernel call in OCLAPI.
    Matrix2 *res = matMakeMatrix2(m.height, m.width);
    res->data = (double *) malloc(sizeof(double) * res->width * res->height);

    size_t gz[] = { res->height, res->width };
    claRunKernel("matt", 2, gz, NULL,
                 m.data, m.width * m.height, OCLREAD | OCLCPY,
                 m.width, m.height,
                 res->data, res->width * res->height, OCLWRITE | OCLOUT);
    if (claGetError()) return NULL;
    
    return res;
}

// PERF: Write this using GPU acceleration. This would either require using `matT2`,
// PERF: or, implement `matlib.h` and use `MatrixN` `matNAtI`.
Tensor* matTTensor(Tensor m) {
    // Error checking.
    if (!matinit) return NULL;

    // Shift dimensions forward.
    int *dims = (int *) malloc(sizeof(int) * m.ndims);
    dims[0] = m.dimsz[m.ndims - 1];
    for (int i = 1; i < m.ndims; i++) dims[i] = m.dimsz[i - 1];

    Tensor *res = matMakeTensor(m.ndims, dims);
    res->data = (double *) malloc(sizeof(double) * res->literal_size);
    free(dims);

    for (int i = 0; i < res->literal_size; i++) {
        int *r_ind = matNIAt(*res, i);

        int *m_ind = (int *) malloc(sizeof(int) * m.ndims);
        for (int j = 0; j < m.ndims - 1; j++) m_ind[j] = r_ind[j + 1];
        m_ind[m.ndims - 1] = r_ind[0];

        *matNAtI(*res, r_ind) = *matNAtI(m, m_ind);

        free(r_ind);
        free(m_ind);
    }

    return res;
}

