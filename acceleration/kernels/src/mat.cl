#ifndef _OCLAPI_INCLUDE_SOURCE
#define _OCLAPI_INCLUDE_SOURCE(source) source
#endif
_OCLAPI_INCLUDE_SOURCE(
__kernel void matmul(__global double *a, __global double *b, unsigned int aw, unsigned int ah, unsigned int bw, __global double *res) {
	int gw = get_global_id(0);
	int gh = get_global_id(1);

	double acc = 0;
	for (int i = 0; i < ah; i++) {
		acc += a[gw + i * aw] * b[i + gh * bw];
	}

	res[gw + gh * aw] = acc;
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
)