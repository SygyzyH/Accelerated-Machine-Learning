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

Tensor* matMakeTensor(unsigned ndims, unsigned *dims, MatrixErr *e) {
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

Tensor* matMakeTensorScalar(double s) {
    Tensor *r = matMakeTensor(1, (unsigned []) { 1 }, NULL);
    r->data = (double *) malloc(sizeof(double));
    r->data[0] = s;

    return r;
}

Tensor* matTensorDeepCopy(Tensor *t, MatrixErr *e) {
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

MatrixErr matTensorFit(Tensor *t1, Tensor *t2, Tensor **t1r, Tensor **t2r) {
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

// Get a refrance to the value of a matrix at a given index.
double* matTensorAtI(Tensor *t, unsigned *ind, MatrixErr *e) {
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

unsigned *matTensorIAt(Tensor *t, int literal, MatrixErr *e) {
    {
        MatrixErr err;
        if (matCheckTensor(t, &err) != MAT_NO_ERROR) {
            if (e != NULL) *e = err;
            return NULL;
        }

        if (literal >= t->literal_size) {
            if (e != NULL) *e = err;
            return NULL;
        }
    }

    unsigned *ind = (unsigned *) malloc(sizeof(unsigned) * t->ndims);
    for (int i = 0; i < t->ndims; i++) {
        ind[i] = (literal / t->dimsz[i]) % t->dimsz[i];
        literal -= ind[i];
    }

    if (e != NULL) *e = MAT_NO_ERROR;
    return ind;
}

// PERF: implement GPU.
// Standardized mul operator for any two tensors.
// Iterate over the first dimension of the first tensor and the second dimension of the
// second tensor. Sum their product.
// The result is sized t1.dims[1:end]t2.dims[0,2:end]. 
// If the resulting dimension is 0 than a scalar is returned.
/*MatrixErr matDot(Tensor t1, Tensor t2, Tensor **r) {
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

// PERF: implement GPU.
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

// PERF: implement GPU.
MatrixErr matAddT(Tensor t1, Tensor t2, Tensor **r) {
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

        *matNAtI(*res, indr) = *matNAtI(*new_t1, ind1) + *matNAtI(*new_t2, ind2);
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
}*/

