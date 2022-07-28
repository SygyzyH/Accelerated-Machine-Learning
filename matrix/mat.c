#include "mat.h"
#include "../acceleration/oclapi.h"
#include <acceleration/kernels/static_kernels_src.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define MAX(a, b) (((a) > (b))? (a) : (b))
#define MIN(a, b) (((a) < (b))? (a) : (b))

bool matinit = false;

MatrixErr matInit() {
    if (matinit) return MAT_NO_ERROR;

    // Source code defined in "acceleration/kernels/static_kernels_src.h"
    const char *src_kernel = KERNEL_STATIC_SOURCE_MAT_CL;
    
    claRegisterFromSrc(&src_kernel, 6, "matmul", "matadd", "matsub", "matprod", "matdot", "sum");
    if (claGetError(1)) return MAT_INITIALIZATION_FAILED;
    
    matinit = true;
    
    return MAT_NO_ERROR;
}

Tensor* matMakeTensor(unsigned ndims, unsigned *dims, MatrixErr *e) {
    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    
    t->ndims = ndims;

    unsigned *dimsz = ndims? (unsigned *) malloc(sizeof(unsigned) * ndims) :\
                      (unsigned *) malloc(sizeof(unsigned));
    if (ndims == 0) dimsz[0] = 1;

    int literal_size = 1;
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

void matTensorPrint(Tensor *t) {
    {
        MatrixErr err;
        if (matCheckTensor(t, &err) != MAT_NO_ERROR) {
            printf_m("Bad Tensor: %s\n", matGetErrorString(err));
            return;
        }
    }
    
    unsigned *ind = calloc(t->ndims, sizeof(unsigned));
    // For any even dimension, print linearly. Including 0.
    // For any odd dimension, print newline.
    
    int even_iter = 1;
    for (int i = 0; i < t->ndims; i += 2) even_iter *= t->dimsz[i];
    int odd_iter = 1;
    for (int i = 1; i < t->ndims; i += 2) odd_iter *= t->dimsz[i];

    for (int odd = 0; odd < odd_iter; odd++) {
        // Accounting for the initial `│`
        int printed_even = -2;

        for (int i = 0; i < t->ndims; i += 2) {
            // This has to be done manually, since the `│` character is extended ASCII.
            printed_even += 2;
            printf("│ ");
        }

        for (int even = 0; even < even_iter; even++) {
            printed_even += printf("%#.6g%s", *matTensorAtI(t, ind, NULL), (ind[0] == t->dimsz[0] - 1)? "" : ", ");
            
            // Increment and carry index.
            ind[0]++;
            for (int i = 0; i < t->ndims; i += 2) {
                if (ind[i] >= t->dimsz[i]) {
                    printed_even += 2;
                    printf("│ ");
                    ind[i] = 0;
                    if (i + 2 < t->ndims) ind[i + 2]++;
                }
            }
        } puts("");

        if (t->ndims > 1) ind[1]++;
        for (int i = 1; i < t->ndims; i+= 2) {
            if (ind[i] >= t->dimsz[i]) {
                printf("├");
                for (int j = 0; j < printed_even - 1; j++) printf("─");
                puts("┤");
                ind[i] = 0;
                if (i + 2 < t->ndims) ind[i + 2]++;
            }
        }
    }

    printf("dims ");
    if (t->ndims == 0) printf("1\n");
    else {
        for (int i = 0; i < t->ndims - 1; i++) printf("%dx", t->dimsz[i]);
        printf("%d\n", t->dimsz[t->ndims - 1]);
    }
}

Tensor* matTensorDeepCopy(Tensor *t, MatrixErr *e) {
    if (t == NULL) {
        if (e != NULL) *e = MAT_NULL_PTR;
        
        return NULL;
    }

    Tensor *r;
    if (matIsTensorScalar(t))
        r = matMakeScalar(t->data[0], e);
    else {
        r = matMakeTensor(t->ndims, t->dimsz, e);
        r->data = (double *) malloc(sizeof(double) * r->literal_size);
        memcpy((void *) r->data, (void *) t->data, t->literal_size * sizeof(double));
    }

    if (e != NULL) *e = MAT_NO_ERROR;

    return r;
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
    
    for (int i = 0; i < biggest->ndims; i++) {
        if (i < t1->ndims)
            if (t1->dimsz[i] != 1)
                t1_dims[i] = t1->dimsz[i];
            else
                t1_dims[i] = t2->dimsz[i];
        else t1_dims[i] = t2->dimsz[i];
    }

    for (int i = 0; i < biggest->ndims; i++) {
        if (i < t2->ndims)
            if (t2->dimsz[i] != 1)
                t2_dims[i] = t2->dimsz[i];
            else 
                t2_dims[i] = t1->dimsz[i];
        else t2_dims[i] = t1->dimsz[i];
    }

    for (int i = 0; i < biggest->ndims; i++) {
        if (t1_dims[i] != t2_dims[i] && t1_dims[i] != 1 && t2_dims[i] != 1) {
            free(t1_dims);
            free(t2_dims);

            return MAT_UNFIT_TENSORS;
        }
    }

    Tensor *t1_res = matMakeTensor(biggest->ndims, t1_dims, NULL);
    free(t1_dims);
    t1_res->data = (double *) malloc(sizeof(double) * t1_res->literal_size);

    for (int i = 0; i < t1_res->literal_size; i++) {
        unsigned *ind = matTensorIAt(t1_res, i, NULL);
        unsigned *ind_org = (unsigned *) malloc(sizeof(unsigned) * t1->ndims);
        
        memcpy(ind_org, ind, t1->ndims * sizeof(unsigned));
        
        for (int sind = 0; sind < t1->ndims; sind++) 
            if (ind[sind] >= t1->dimsz[sind])
                ind_org[sind] %= t1->dimsz[sind];
        
        *matTensorAtI(t1_res, ind, NULL) = *matTensorAtI(t1, ind_org, NULL);

        free(ind);
        free(ind_org);
    }
    
    Tensor *t2_res = matMakeTensor(biggest->ndims, t2_dims, NULL);
    free(t2_dims);
    t2_res->data = (double *) malloc(sizeof(double) * t2_res->literal_size);

    for (int i = 0; i < t2_res->literal_size; i++) {
        unsigned *ind = matTensorIAt(t2_res, i, NULL);
        unsigned *ind_org = (unsigned *) malloc(sizeof(unsigned) * t2->ndims);
        memcpy(ind_org, ind, sizeof(unsigned) * t2->ndims);
        
        for (int sind = 0; sind < t2->ndims; sind++) 
            if (ind[sind] >= t2->dimsz[sind])
                ind_org[sind] = ind[sind] % t2->dimsz[sind];
 
        *matTensorAtI(t2_res, ind, NULL) = *matTensorAtI(t2, ind_org, NULL);

        free(ind);
        free(ind_org);
    }

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

    if (matIsTensorScalar(t)) {
        if (e != NULL) *e = MAT_NO_ERROR;
        return t->data;
    }

    int r = 0;
    int mult = 1;

    for (int i = 0; i < t->ndims; i++) {
        r += ind[i] * mult;
        mult *= t->dimsz[i];
    }

    if (r > t->literal_size) {
        if (e != NULL) *e = MAT_DIMENSION_OUT_OF_RANGE;
        return NULL;
    }

    if (e != NULL) *e = MAT_NO_ERROR;
    return (double *) (t->data + r);
}

unsigned* matTensorIAt(Tensor *t, int literal, MatrixErr *e) {
    {
        if (matCheckTensor(t, e) != MAT_NO_ERROR) return NULL;

        if (literal >= t->literal_size) {
            if (e != NULL) *e = MAT_DIMENSION_OUT_OF_RANGE;
            return NULL;
        }
    }

    unsigned *ind = (unsigned *) malloc(sizeof(unsigned) * t->ndims);
    unsigned stride = 1;
    for (int i = 0; i < t->ndims; i++) {
        ind[i] = (literal / stride) % t->dimsz[i];
        stride *= t->dimsz[i];
    }

    if (e != NULL) *e = MAT_NO_ERROR;
    return ind;
}

MatrixErr matSum(double *src, int size, double *res) {
    if (src == NULL) return MAT_NULL_PTR;
    if (res == NULL) return MAT_NULL_PTR;

    size_t gz[] = { size };
    claRunKernel("sum", 1, gz, NULL,
                 src, size, OCLREAD | OCLCPY,
                 size,
                 NULL, size, OCLWRITE | OCLREAD,
                 res, 1, OCLWRITE | OCLOUT);
    if (claGetError(1)) return MAT_KERNEL_FAILURE;

    return MAT_NO_ERROR;
}

MatrixErr matProd(Tensor *t1, Tensor *t2, Tensor **r) {
    if (r == NULL) return MAT_NULL_PTR;
    *r = NULL;
    {
        MatrixErr err;
        if (matCheckTensor(t1, &err) != MAT_NO_ERROR) return err;
        if (matCheckTensor(t2, &err) != MAT_NO_ERROR) return err;
    }

    if (t1->ndims == 0 || t2->ndims == 0) return MAT_DIMENSION_MISTMATCH;
    if (t1->ndims != 1 && t2->ndims != 1 && t1->ndims != t2->ndims) return MAT_DIMENSION_MISTMATCH;
    
    int t1_vector = 0;
    unsigned *odimsz1 = t1->dimsz;
    if (t1->ndims == 1) {
        t1->ndims = MAX(2, t2->ndims);
        odimsz1 = t1->dimsz;
        t1->dimsz = (unsigned *) malloc(sizeof(unsigned) * t1->ndims);
        t1->dimsz[0] = odimsz1[0];
        for (int i = 1; i < t1->ndims; i++) t1->dimsz[i] = 1;
        
        t1_vector = 1;
    }
    
    int t2_vector = 0;
    unsigned *odimsz2 = t2->dimsz;
    if (t2->ndims == 1) {
        t2->ndims = MAX(2, t1->ndims);
        odimsz2 = t2->dimsz;
        t2->dimsz = (unsigned *) malloc(sizeof(unsigned) * t2->ndims);
        t2->dimsz[0] = 1;
        t2->dimsz[1] = odimsz2[0];
        for (int i = 2; i < t2->ndims; i++) t2->dimsz[i] = 1;
        
        t2_vector = 1;
    }
    
    if (t1->dimsz[0] != ((t2->ndims > 1)? t2->dimsz[1] : t2->dimsz[0])) return MAT_UNFIT_TENSORS;

    Tensor *biggest = (t1->ndims > t2->ndims)? t1 : t2;
    int rndims = biggest->ndims;
    unsigned *rdimsz = (unsigned *) malloc(sizeof(unsigned) * rndims);
    for (int i = 2; i < rndims; i++) {
        if (t1->dimsz[i] != t2->dimsz[i] && t1->dimsz[i] != 1 && t2->dimsz[i] != 1)
            return MAT_UNFIT_TENSORS;
        
        rdimsz[i] = biggest->dimsz[i];
    }
    rdimsz[0] = t2->dimsz[0];
    rdimsz[1] = (t1->ndims > 1)? t1->dimsz[1] : t1->dimsz[0];

    // Standard kernel call in OCLAPI.
    *r = matMakeTensor(rndims, rdimsz, NULL);
    free(rdimsz);
    Tensor *res = *r;
    res->data = (double *) malloc(sizeof(double) * res->literal_size);

    size_t gz[] = { res->literal_size };
    claRunKernel("matprod", 1, gz, NULL,
                 t1->data, t1->literal_size, OCLREAD | OCLCPY,
                 t2->data, t2->literal_size, OCLREAD | OCLCPY,
                 t1->dimsz, t1->ndims, OCLREAD | OCLCPY,
                 t2->dimsz, t2->ndims, OCLREAD | OCLCPY,
                 res->data, res->literal_size, OCLWRITE | OCLOUT,
                 res->dimsz, res->ndims, OCLREAD | OCLCPY,
                 res->ndims);

    if (t1_vector) {
        t1->ndims = 1;
        free(t1->dimsz);
        t1->dimsz = odimsz1;
    }
    
    if (t2_vector) {
        t2->ndims = 1;
        free(t2->dimsz);
        t2->dimsz = odimsz2;
    }

    if (claGetError(1)) {
        matFreeTensor(r);
        return MAT_KERNEL_FAILURE;
    }

    if (matIsTensorScalar(res)) {
        double res_s = res->data[0];
        matFreeTensor(r);
        *r = matMakeScalar(res_s, NULL);
    }

    matTensorReduce(*r);

    return MAT_NO_ERROR;
}

MatrixErr matDot(Tensor *t1, Tensor *t2, Tensor **r) {
    if (r == NULL) return MAT_NULL_PTR;
    *r = NULL;
    {
        MatrixErr err;
        if (matCheckTensor(t1, &err) != MAT_NO_ERROR) return err;
        if (matCheckTensor(t2, &err) != MAT_NO_ERROR) return err;
    }
    
    if (matIsTensorScalar(t1) || matIsTensorScalar(t2)) return matMult(t1, t2, r);
    if (t1->ndims <= 2 && t2->ndims <= 2) return matProd(t1, t2, r);
    
    // a Tensors is nD, n > 2.
    // Check if first dimension of t1 equals the second dimension of t2.
    if (t1->dimsz[0] != t2->dimsz[MIN(1, t2->ndims - 1)]) return MAT_DIMENSION_MISTMATCH;

    unsigned ndims = MAX(0, t1->ndims + t2->ndims - 2);
    unsigned *dimsz = (unsigned *) malloc(sizeof(unsigned) * (ndims? ndims : 1));
    // For scalar result. If nonscalar - this would be overwritten.
    dimsz[0] = 1;
    for (int i = 0; i < ndims; i++) {
        if (i + 1 < t1->ndims) dimsz[i] = t1->dimsz[i + 1];
        else if (i + 1 == t1->ndims) dimsz[i] = t2->dimsz[0];
        else dimsz[i] = t2->dimsz[MAX(i - t1->ndims + 2, t2->ndims - 1)];
    }

    *r = matMakeTensor(ndims, dimsz, NULL);
    free(dimsz);
    Tensor *res = *r;
    res->data = (double *) malloc(sizeof(double) * res->literal_size);

    size_t gz[] = { res->literal_size };
    claRunKernel("matdot", 1, gz, NULL,
                 t1->data, t1->literal_size, OCLREAD | OCLCPY,
                 t2->data, t2->literal_size, OCLREAD | OCLCPY,
                 t1->ndims, t1->dimsz, t1->ndims, OCLREAD | OCLCPY,
                 t2->ndims, t2->dimsz, t2->ndims, OCLREAD | OCLCPY,
                 res->data, res->literal_size, OCLWRITE | OCLOUT,
                 res->ndims, res->dimsz, res->ndims, OCLREAD | OCLCPY);
    if (claGetError(1)) {
        matFreeTensor(r);
        return MAT_KERNEL_FAILURE;
    }

    matTensorReduce(*r);

    return MAT_NO_ERROR;
}

MatrixErr _matSTDLinearCall(Tensor *t1, Tensor *t2, Tensor **r, const char *kname);

MatrixErr _matSTDLinearCall(Tensor *t1, Tensor *t2, Tensor **r, const char *kname) {
    if (r == NULL) return MAT_NULL_PTR;
    *r = NULL;
    {
        MatrixErr err;
        if (matCheckTensor(t1, &err) != MAT_NO_ERROR) return err;
        if (matCheckTensor(t2, &err) != MAT_NO_ERROR) return err;
    }

    Tensor *new_t1;
    Tensor *new_t2;
    if (matTensorFit(t1, t2, &new_t1, &new_t2)) return MAT_UNFIT_TENSORS;

    // Standard kernel call in OCLAPI.
    unsigned *rdimsz = (unsigned *) malloc(sizeof(unsigned) * new_t1->ndims);
    for (int i = 0; i < new_t2->ndims; i++) 
        rdimsz[i] = new_t2->dimsz[i];
    rdimsz[0] = new_t1->dimsz[0];

    *r = matMakeTensor(new_t1->ndims, rdimsz, NULL);
    free(rdimsz);
    Tensor *res = *r;
    res->data = (double *) malloc(sizeof(double) * res->literal_size);

    size_t gz[] = { res->literal_size };
    claRunKernel(kname, 1, gz, NULL,
                 new_t1->data, new_t1->literal_size, OCLREAD | OCLCPY,
                 new_t2->data, new_t2->literal_size, OCLREAD | OCLCPY,
                 res->data, res->literal_size, OCLWRITE | OCLOUT);
    if (claGetError(1)) { 
        matFreeTensor(r);
        return MAT_KERNEL_FAILURE;
    }

    return MAT_NO_ERROR;
}

inline MatrixErr matAdd(Tensor *t1, Tensor *t2, Tensor **r) {
    return _matSTDLinearCall(t1, t2, r, "matadd");
}

inline MatrixErr matSub(Tensor *t1, Tensor *t2, Tensor **r) {
    return _matSTDLinearCall(t1, t2, r, "matsub");
}

inline MatrixErr matMult(Tensor *t1, Tensor *t2, Tensor **r) {
    return _matSTDLinearCall(t1, t2, r, "matmul");
}

// PERF: LONG TERM : making a "numpy view"-like implementation, just like previus commits.
MatrixErr matTTensor(Tensor *t, Tensor **r) {
    {
        MatrixErr err;
        if (matCheckTensor(t, &err)) return err;
    }
    if (r == NULL) return MAT_NULL_PTR;
    *r = NULL;

    unsigned *dimsz = (unsigned *) malloc(sizeof(unsigned) * t->ndims);
    dimsz[0] = t->dimsz[t->ndims - 1];
    for (int i = 1; i < t->ndims; i++) dimsz[i] = t->dimsz[i - 1];

    Tensor *res = matMakeTensor(t->ndims, dimsz, NULL);
    res->data = (double *) malloc(sizeof(double) * res->literal_size);
    free(dimsz);

    // Copy the data, offset.
    for (int i = 0; i < t->literal_size; i++) {
        unsigned *ind = matTensorIAt(t, i, NULL);

        unsigned last = ind[t->ndims - 1];
        for (int j = t->ndims; j --> 0;) ind[j] = ind[j - 1];
        ind[0] = last;
        
        *matTensorAtI(res, ind, NULL) = t->data[i];

        free(ind);
    }

    *r = res;

    return MAT_NO_ERROR;
}
