#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "matrix/mat.h"
#include "acceleration/oclapi.h"
#include "ml/ml.h"
#include <acceleration/kernels/static_kernels_src.h>

#include <assert.h>
int main() {
    int error;

    error = claInit();
    printf("err: %s\n", claGetErrorString(error));
    printf("oclapi internal: %s\n", clGetErrorString(claGetExtendedError()));

    error = matInit();

    // MatrixN *print_test = matMakeMatrixN(6, (int []) { 2, 2, 2, 2, 2, 2, 2, 2 });
    // print_test->data = (double *) malloc(sizeof(double) * print_test->literal_size);
    // for (int i = 0; i < print_test->literal_size; i++) print_test->data[i] = i;
    // matPrintMatrixN(*print_test);
    
    Matrix2 *l1w = matMakeMatrix2(2, 2);
    Matrix2 *l2w = matMakeMatrix2(1, 2);
    l1w->data = (double []) { 0.11, 0.12, 0.21, 0.08 };
    l2w->data = (double []) { 0.14, 0.15 };

    Machine m = mlMakeMachine(3, (Layer* []) { 
                      mlMakeLayer(FullyConnected, NULL, l1w),
                      mlMakeLayer(FullyConnected, NULL, l2w),
                      mlMakeLayer(MeanSquaredError, NULL, NULL)
                });

    Tensor *inp = matMakeTensor(1, (int []) { 2 });
    inp->data = (double []) { 2, 3 };

    Tensor *out = NULL;

    error = mlMachineFeedForward(m, inp, &out);
    fprintf(stderr, "error: %s\n", mlGetErrorString(error));

    assert(out != NULL);
    matPrintTensor(*out);

    Tensor *desired_output = matMakeTensor(2, (int []) { 1, 1 });
    desired_output->data = (double []) { 1 };

    double learning_rate = 0.05;
    LearningInstance *inst = mlMakeLearningInstance(m, (void *) &learning_rate, 1, inp, desired_output, SGD);
    assert(inst != NULL);
    error = mlTrainInstance(inst);
    printf("training error: %s\n", mlGetErrorString(error));

    /*Tensor *o2 = NULL;
    mlMachineFeedForward(&m, inp, &o2);

    assert(o2 != NULL);
    matPrintTensor(*out);*/
    //matPrintMatrix2(*(Matrix2 *)m.layers[0]->weights);
 
    claCln();
    
    return 0;
}
