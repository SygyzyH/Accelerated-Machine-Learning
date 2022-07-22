#include "ml.h"
#include "../matrix/mat.h"
#include <math.h>
#include <stdlib.h>
// NOTE: Random number generation uses time.
#include <time.h>

MLErr mlMachineFeedForward(Machine machine, Tensor *input, Tensor **output) {
    if (matCheckTensor(input, NULL)) return ML_MAT_ERROR;
    
    if (output == NULL) return ML_NULL_PTR;
    *output = NULL;

    Tensor *current_inp = input;
    Tensor *current_output = NULL;

    for (int layeri = 0; layeri < machine.layer_count; layeri++) {
        MLErr error = machine.layers[layeri]->forward(machine.layers[layeri], *current_inp, &current_output);
        printf("for index #%d:\n", layeri);
        matTensorPrint(current_output);
        if (current_inp != input) matFreeTensor(&current_inp);
        if (error != ML_NO_ERR) {
            matFreeTensor(&current_output);
            current_output = NULL;
            return error;
        }

        current_inp = current_output;
    }

    *output = current_output;

    return ML_NO_ERR;
}

// TODO: Is this function usefull? Shouldn't each layer implement its weight initializer?
// Or maybe this function is usefull to be used inside the implementation?
// TODO: A function with the exact same functunality should be added to mat lib, and this one either acting as an API call to it or removing the function entirely.
Tensor* mlWeightInitializer(MLWeightInitializerType initializer, unsigned ndims, unsigned *dims) {
    Tensor *res = matMakeTensor(ndims, dims, NULL);
    res->data = (double *) malloc(sizeof(double) * res->literal_size);

    switch (initializer) {
        case ML_WEIGHT_INITIALIZER_ZEROS: {
            for (int i = 0; i < res->literal_size; i++) res->data[i] = 0;
            break;
        }
        
        case ML_WEIGHT_INITIALIZER_ONES: {
            for (int i = 0; i < res->literal_size; i++) res->data[i] = 1;
            break;
        }

        case ML_WEIGHT_INITIALIZER_GLOROT: {
            double start = -1.0 / sqrt(res->literal_size);
            double end = -start;
            double step = end - start;
            
            srand(time(NULL));

            for (int i = 0; i < res->literal_size; i++) res->data[i] = start + ((double) rand() / RAND_MAX) / step;
            matTensorPrint(res);
            break;
        }

        default: {
            matFreeTensor(&res);
            return NULL;
        }
    }

    return res;
}
