// TODO: Kernels should use stride instead of width for indexing. Can be done in include.
#include <opencl-c-base.h>
#include <opencl-c.h>

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

__kernel void sum(__global double *src, int src_size, __local double *temp, __global double *res) {
    int li = get_global_id(0);
    sumArray(&temp, src_size, li);
    if (li == 0) res[0] = temp[0];
}

unsigned remapLinearIndexSpace(int literal, __global unsigned *source_mapping, __global unsigned *target_mapping, int mapping_size);

// assert(target_mapping_size == source_mapping_size)
unsigned remapLinearIndexSpace(int literal, __global unsigned *source_mapping, __global unsigned *target_mapping, int mapping_size) {
    unsigned sum = 0;
    unsigned source_stride = 1;
    unsigned target_stride = 1;

    for (int i = 0; i < mapping_size; i++) {
        int ind = (literal / source_stride) % source_mapping[i];
        source_stride *= source_mapping[i];
        if (target_mapping[i] > 1) sum += ind * target_stride;
        target_stride *= target_mapping[i];
    }
    
    return sum;
}

// adimsz = adimsz[0], bdimsz = bdimsz[0]
__kernel void matprod(__global double *a, __global double *b, __global unsigned *adimsz, __global unsigned *bdimsz, __global double *r, __global unsigned *rdimsz, int ndims) {
    int gi = get_global_id(0);

    unsigned a_stride = 1;
    unsigned b_stride = bdimsz[0];

    int gi_dim0 = gi % rdimsz[0];
    int offseta = remapLinearIndexSpace(gi - gi_dim0, rdimsz, adimsz, ndims);
    
    // gi excluding first dimension index
    int gi_dim1 = ((gi - gi_dim0) / rdimsz[0]);
    if (ndims > 1) gi_dim1 %= rdimsz[1];
    int offsetb = remapLinearIndexSpace(gi - gi_dim1 * rdimsz[0], rdimsz, bdimsz, ndims);

    // assert(bdimsz[1] == adimsz[0])
    // Common dimension.
    int iter = adimsz[0];

    r[gi] = 0;
    for (int i = 0; i < iter; i++) {
        //printf("gi: #%d, i: #%d, offseta: %d, a: %d, offsetb: %d, b: %d\\n",
        //        gi, i, offseta, offseta + i * a_stride, offsetb, offsetb + i * b_stride);
        r[gi] += a[offseta + i * a_stride] * b[offsetb + i * b_stride];
    }
}

__kernel void matdot(__global double *a, __global double *b, unsigned andims, __global unsigned *adimsz, unsigned bndims, __global unsigned *bdimsz, __global double *r, unsigned rndims, __global unsigned *rdimsz) {
    int gi = get_global_id(0);

    unsigned a_stride = 1;
    unsigned b_stride = bdimsz[0];

    // is first andims - 1 indecies of rndims
    unsigned a_ind = remapLinearIndexSpace(gi, rdimsz, &adimsz[1], andims - 1) * adimsz[0];
    
    unsigned a_mul_st = 1;
    for (int i = 1; i < andims; i++) a_mul_st *= adimsz[i];
    unsigned dimba = (gi - a_ind / adimsz[0]) / a_mul_st;
    unsigned fdimb = dimba % bdimsz[0];
    unsigned b_ind = fdimb + (dimba - fdimb) * bdimsz[1];

    r[gi] = 0;
    for (int i = 0; i < adimsz[0]; i++)
        r[gi] += a[a_ind + a_stride * i] * b[b_ind + b_stride * i];
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
