#include "mat.h"
#include "../acceleration/oclapi.h"
#include <acceleration/kernels/static_kernels_src.h>
#include <stdbool.h>

bool matinit = false;

MatrixErr matInit() {
    const char *src_kernel = KERNEL_STATIC_SOURCE_MAT_CL;
    
    claRegisterFromSrc(&src_kernel, 5, "matmul", "matadd", "matsub", "matmuls", "matadds");
    if (claGetError()) return MAT_INITIALIZATION_FAILED;
    
    matinit = true;
    
    return MAT_NO_ERROR;
}

MatrixErr matMul(Matrix2 m1, Matrix2 m2, Matrix2 **r) {
    *r = NULL;
    if (!matinit) return MAT_UNINITIALIZED;
    if (m1.width != m2.height) return MAT_DIMENSION_MISTMATCH;
    
    *r = makeMatrix2(m2.width, m1.height);
    Matrix2 *res = *r;
    res->data = (double *) malloc(sizeof(double) * res->width * res->height);

    size_t gz[] = { res->width, res->height };
    claRunKernel("matmul", 2, gz, NULL,
                 m1.data, m1.height * m1.stride, OCLREAD | OCLCPY,
                 m2.data, m2.height * m2.stride, OCLREAD | OCLCPY,
                 m2.width, m1.width, 
                 res->data, res->width * res->height, OCLWRITE | OCLOUT);
    if (claGetError()) return MAT_KERNEL_FAILURE;

    return MAT_NO_ERROR;
}

MatrixErr matAdd(Matrix2 m1, Matrix2 m2, Matrix2 **r) {
    *r = NULL;
    if (!matinit) return MAT_UNINITIALIZED;
    if (m1.width != m2.width || m1.height != m2.height) return MAT_DIMENSION_MISTMATCH;
    
    *r = makeMatrix2(m1.width, m1.height);
    Matrix2 *res = *r;
    res->data = (double *) malloc(sizeof(double) * res->height * res->stride);

    size_t gz[] = { res->width, res->height };
    claRunKernel("matadd", 2, gz, NULL, 
                 m1.data, m1.width * m1.height, OCLREAD | OCLCPY,
                 m2.data, m2.width * m2.height, OCLREAD | OCLCPY,
                 m1.width, m1.height,
                 res->data, res->height * res->stride, OCLWRITE | OCLOUT);
    if (claGetError()) return MAT_KERNEL_FAILURE;

    return MAT_NO_ERROR;
}
