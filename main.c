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

    Matrix2 *l1w = matMakeMatrix2(2, 2);
    Matrix2 *l2w = matMakeMatrix2(1, 2);
    l1w->data = (double []) { 0.11, 0.12, 0.21, 0.08 };
    l2w->data = (double []) { 0.14, 0.15 };

    Machine m = mlMakeMachine(4, (Layer* []) { 
                      mlMakeLayer(FullyConnected, NULL, matTensorAsMatrix2(*mlWeightInitializer(ML_WEIGHT_INITIALIZER_GLOROT, 2, (int []) { 2, 2 }), 2, 2)),
                      mlMakeLayer(FullyConnected, NULL, matTensorAsMatrix2(*mlWeightInitializer(ML_WEIGHT_INITIALIZER_GLOROT, 2, (int []) { 1, 2 }), 1, 2)),
                      mlMakeLayer(Bias, NULL, mlWeightInitializer(ML_WEIGHT_INITIALIZER_ZEROS, 1, (int []) { 1 })),
                      mlMakeLayer(MeanSquaredError, NULL, NULL)
                });

    Tensor *inp = matMakeTensor(1, (int []) { 2 });
    inp->data = (double []) { 2, 3 };

    Tensor *out = NULL;

    puts("Feed forward");
    error = mlMachineFeedForward(m, inp, &out);

    assert(out != NULL);
    matPrintTensor(*out);

    Tensor *desired_output = matMakeTensorScalar(1);

    double learning_rate = 0.05;
    LearningInstance *inst = mlMakeLearningInstance(m, (void *) &learning_rate, 1, inp, desired_output, SGD);
    assert(inst != NULL);
    puts("Training");
    
    const int training_rep = 60;
    for (int i = 0; i < training_rep; i++) error = mlTrainInstance(inst);

    printf("training error: %s\n", mlGetErrorString(error));

    Tensor *o2 = NULL;
    mlMachineFeedForward(m, inp, &o2);

    assert(o2 != NULL);
    matPrintTensor(*o2);
 
    claCln();
    
    return 0;
}
