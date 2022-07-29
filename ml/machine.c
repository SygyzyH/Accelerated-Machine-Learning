#include "ml.h"
#include "../matrix/mat.h"

#include <stdlib.h>

MLErr mlMachineFeedForward(Machine machine, Tensor *input, Tensor **output) {
    if (matCheckTensor(input, NULL)) return ML_MAT_ERROR;
    
    if (output == NULL) return ML_NULL_PTR;
    *output = NULL;

    Tensor *current_inp = input;
    Tensor *current_output = NULL;

    for (int layeri = 0; layeri < machine.layer_count; layeri++) {
        MLErr error = machine.layers[layeri]->forward(machine.layers[layeri], current_inp, &current_output);
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

