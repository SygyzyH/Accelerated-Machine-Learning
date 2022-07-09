// TODO: Kernels should use stride instead of width for indexing. Can be done in include.
#include <opencl-c-base.h>
#include <opencl-c.h>
//#include "../include/matutil.h"

void sumArray(__local double **tmp, int bsize, int li);

void sumArray(__local double **tmp, int bsize, int li) {
    // Ensure input is ready before function runs
    barrier(CLK_LOCAL_MEM_FENCE);
    __local double *temp = *tmp;
    
    int halfbsize = (int) bsize / 2;
    while (halfbsize > 0) {
        if (li < halfbsize) {
            temp[li] += temp[li + halfbsize];
            if (bsize % 2 == 1 && li == 0)
                temp[li] += temp[li + bsize - 1];
        }
        
        // Proceed to next half
        barrier(CLK_LOCAL_MEM_FENCE);
        bsize = halfbsize;
        halfbsize = (int) bsize / 2;
    }
    
    // Ensure output is ready once the function returns
    barrier(CLK_LOCAL_MEM_FENCE);
}

// adimsz = adimsz[0], bdimsz = bdimsz[0]
__kernel void matprod(__global double *a, __global double *b, unsigned adimsz, unsigned bdimsz, __global double *r, __global unsigned *rdimsz) {
    // TODO: Support for fitted tensors
    int gi = get_global_id(0);

    int a_stride = 1;
    int b_stride = bdimsz;

    int offseta = gi - (gi % rdimsz[0]);
    // gi ecluding first dimension index
    int offsetb = gi - ((offseta / rdimsz[1]) % rdimsz[1]) * rdimsz[1];

    // assert(bdimsz[1] == adimsz[0])
    int iter = adimsz;

    r[gi] = 0;
    for (int i = 0; i < iter; i++) {
        r[gi] += a[offseta + i * a_stride] * b[offsetb + i * b_stride];
    }
}

__kernel void matadd(__global double *a, __global double *b, __global double *r) {
    int gi = get_global_id(0);
    r[gi] = a[gi] + b[gi];
}

__kernel void matsub(__global double *a, __global double *b, __global double *r) {
    int gi = get_global_id(0);
    r[gi] = a[gi] - b[gi];
}

__kernel void matmul(__global double *a, __global double *b, __global double *r) {
    int gi = get_global_id(0);
    r[gi] = a[gi] * b[gi];
}
