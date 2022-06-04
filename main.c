#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "matrix/mat.h"
#include "acceleration/oclapi.h"
#include "ml/ml.h"
#include <acceleration/kernels/static_kernels_src.h>

int main() {
    int error;

    error = claInit();
    printf("err: %s\n", claGetErrorString(error));
    printf("oclapi internal: %s\n", clGetErrorString(claGetExtendedError()));

    error = matInit();
    
    MatrixN *l1w = makeMatrixN(2, (int []) { 2, 2 });
    MatrixN *l2w = makeMatrixN(2, (int []) { 1, 2 });
    l1w->data = (double []) { 0.11, 0.12, 0.21, 0.08 };
    l2w->data = (double []) { 0.14, 0.15 };
    Layer *l1 = mlMakeLayer(*l1w, mlFullyConnectedLayerForward, mlFullyConnectedLayerGetDelta);
    Layer *l2 = mlMakeLayer(*l2w, mlFullyConnectedLayerForward, mlFullyConnectedLayerGetDelta);

    Machine machine;
    machine.layers = NULL;
    mlAddLayer(&machine, l1);
    mlAddLayer(&machine, l2);

    MatrixN *machine_input = makeMatrixN(2, (int []) { 1, 2 });
    MatrixN *machine_output;
    machine_input->data = (double []) { 2, 3 };
    mlFeedForward(machine, *machine_input, &machine_output);
    // machine_output = makeMatrixN(2, (int []) { 1, 1 });
    // machine_output->data = (double []) { 2 };

    matPrintMatrix2(*matNAsMatrix2(*machine_output, 1, 1));

    claCln();
    
    return 0;
}
