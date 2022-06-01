#include <stdlib.h>
#include <stdio.h>

#include "acceleration/oclapi.h"
#include "acceleration/kernels/static_kernels_src.h"

int main() {
  OCLAPIErr error;

  error = clainit();
  printf("err: %s\n", claGetErrorString(error));
  printf("oclapi internal: %s\n", clGetErrorString(claGetExtendedError()));

  const char *matsrc = KERNEL_STATIC_SOURCE_MAT_CL;
  error = claRegisterFromSrc(&matsrc, 5, "matmul", "matadd", "matsub", "matmuls",
                         "matadds");
  printf("err: %s\n", claGetErrorString(error));

  double a[] = {1, 1, 1, 1};
  double b[] = {1, 2, 77, 2};
  double *r = malloc(sizeof(double) * 4);
  size_t gz[] = {2, 2};

  error = claRunKernel("matadd", 2, gz, NULL, a, 4, OCLCPY | OCLREAD, b, 4,
                       OCLCPY | OCLREAD, 2, 2, r, 4, OCLOUT | OCLWRITE);
  printf("err: %s\n", claGetErrorString(error));

  puts("result: ");
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      printf("%lf, ", r[j + i * 2]);
    }
    puts("");
    }
    
    return 0;
}
