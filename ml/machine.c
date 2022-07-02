#include "ml.h"
#include "../matrix/mat.h"
#include <stdlib.h>

MLErr mlMachineFeedForward(Machine machine, Tensor *input, Tensor **output) {
    if (output == NULL) return ML_NULL_PTR;
    *output = NULL;

    Tensor *current_inp = input;
    Tensor *current_output = NULL;

    for (int layeri = 0; layeri < machine.layer_count; layeri++) {
        MLErr error = machine.layers[layeri]->forward(machine.layers[layeri], *current_inp, &current_output);
        if (current_inp != input) freeTensor(current_inp);
        if (error != ML_NO_ERR) {
            freeTensor(current_output);
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
Tensor* mlWeightInitializer(MLWeightInitializerType initializer, int ndims, int *dims) {
    Tensor *res = matMakeTensor(ndims, dims);
    res->data = (double *) malloc(sizeof(double) * res->literal_size);

    switch (initializer) {
        case ML_WEIGHT_INITIALIZER_ZEROS: {
            for (int i = 0; i < res->literal_size; i++) {
                int *ind = matNIAt(*res, i);
                double *d = matNAtI(*res, ind);
                
                *d = 0;

                free(ind);
            }
            break;
        }
        
        case ML_WEIGHT_INITIALIZER_ONES: {
            for (int i = 0; i < res->literal_size; i++) {
                int *ind = matNIAt(*res, i);
                double *d = matNAtI(*res, ind);
                
                *d = 1;

                free(ind);
            }
            break;
        }

        default: {
            freeTensor(res);
            return NULL;
        }
    }

    return res;
}
