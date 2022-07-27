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
    printf("oclapi internal: %s\n", clGetErrorString(claGetExtendedError(1)));

    error = matInit();

    Tensor *l1w = matMakeTensor(2, (unsigned []) { 2, 2 }, NULL);
    Tensor *l2w = matMakeTensor(2, (unsigned []) { 2, 1 }, NULL);
    l1w->data = (double []) { 0.11, 0.21, 0.12, 0.08 };
    l2w->data = (double []) { 0.14, 0.15 };

    Machine m = mlMakeMachine(5, (Layer *[]) {
                                mlMakeLayer(FullyConnected, NULL, l1w),
                                mlMakeLayer(Bias, NULL, mlWeightInitializer(ML_WEIGHT_INITIALIZER_ONES, 1, (unsigned []) { 2 })),
                                mlMakeLayer(FullyConnected, NULL, l2w),
                                mlMakeLayer(Bias, NULL, mlWeightInitializer(ML_WEIGHT_INITIALIZER_ONES, 1, (unsigned []) { 1 })),
                                mlMakeLayer(MeanSquaredError, NULL, NULL)
                            });

    Tensor *inp = matMakeTensor(1, (unsigned []) { 2 }, NULL);
    inp->data = (double []) { 2, 3 };

    Tensor *desired_output = matMakeScalar(1, NULL);
    double learning_rate = 0.05;

    LearningInstance *inst = mlMakeLearningInstance(m, &learning_rate, 1, inp, desired_output, SGD);
    //for (int i = 0; i < 69; i++) 
    printf("terr: %s\n", mlGetErrorString(mlTrainInstance(inst)));
    
    Tensor *res = NULL;

    mlMachineFeedForward(m, inp, &res);

    puts("output:");
    matTensorPrint(res);

    claCln();
    
    return 0;
}
