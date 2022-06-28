#include "ml.h"
#include "../matrix/mat.h"

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
    return NULL;
}
