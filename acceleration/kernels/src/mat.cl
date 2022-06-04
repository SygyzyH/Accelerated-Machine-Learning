// TODO: Kernels should use stride instead of width for indexing.
#include <opencl-c-base.h>
#include "../include/matutil.h"

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

__kernel void matmul(__global double *a, __global double *b, unsigned int bw, unsigned int com, __global double *res) {
    int gw = get_global_id(0);
    int gh = get_global_id(1);

    double acc = 0;
    for (int i = 0; i < com; i++) 
    	acc += a[i + gh * com] * b[gw + i * bw];
    
    res[gw + gh * bw] = acc;
}

__kernel void matadd(__global double *a, __global double *b, unsigned int w, unsigned int h, __global double *res) {
    int gw = get_global_id(0);
    int gh = get_global_id(1);

    res[gw + gh * w] = a[gw + gh * w] + b[gw + gh * w];
}

__kernel void matsub(__global double *a, __global double *b, unsigned int w, unsigned int h, __global double *res) {
    int gw = get_global_id(0);
    int gh = get_global_id(1);

    res[gw + gh * w] = a[gw + gh * w] - b[gw + gh * w];
}

__kernel void matdot(__global double *a, __global double *b, unsigned int w, unsigned int h, __local double *sum_ar, __global double *res) {
    int li = get_local_id(0);
    
    sum_ar[li] = a[li] * b[li];
    int bsize = w * h;

    int halfbsize = (int) bsize / 2;
    while (halfbsize > 0) {
        if (li < halfbsize) {
            sum_ar[li] += sum_ar[li + halfbsize];
            if (bsize % 2 == 1 && li == 0)
                sum_ar[li] += sum_ar[li + bsize - 1];
        }
        
        // Proceed to next half
        barrier(CLK_LOCAL_MEM_FENCE);
        bsize = halfbsize;
        halfbsize = (int) bsize / 2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    *res = sum_ar[0];
}

__kernel void matmuls(__global double *a, double s, unsigned int w, unsigned int h, __global double *res) {
    int gw = get_global_id(0);
    int gh = get_global_id(1);

    res[gw + gh * w] = a[gw + gh * w] * s;
}

__kernel void matadds(__global double *a, double s, unsigned int w, unsigned int h, __global double *res) {
    int gw = get_global_id(0);
    int gh = get_global_id(1);

    res[gw + gh * w] = a[gw + gh * w] + s;
}
