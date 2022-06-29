#include "ml.h"
#include <stdio.h>
#include <stdlib.h>

MLErr mlTrainInstance(LearningInstance *instance) {
    if (!instance->src_machine._all_layers_initialized) return ML_MACHINE_UNINITIALIZED_LAYER;

    // Convinience
    Machine src_machine = instance->src_machine;

    // First, collect the activations.
    // activations is an `input_n x layer_count` array of activations.
    // NOTE: `activations[0] = input` 
    Tensor **activations = (Tensor **) malloc(sizeof(Tensor *) * instance->input_n);
    Tensor **derivatives = (Tensor **) malloc(sizeof(Tensor *) * instance->input_n);
    for (int i = 0; i < instance->input_n; i++)
        activations[i] = (Tensor *) malloc(sizeof(Tensor) * src_machine.layer_count);
    for (int i = 0; i < instance->input_n; i++)
        derivatives[i] = (Tensor *) malloc(sizeof(Tensor) * src_machine.layer_count);

    // Clearer free of activations and derivatives. 
#define freeActivationsDerivatives \
    for (int i = 0; i < instance->input_n; i++) { \
        for (int j = 0; j < src_machine.layer_count; j++) { \
            if (j != 0) freeTensorD(activations[i][j]); \
            freeTensorD(derivatives[i][j]); \
        } \
        free(activations[i]); \
        activations[i] = NULL; \
        free(derivatives[i]); \
        derivatives[i] = NULL; \
    } \
    free(activations); \
    activations = NULL; \
    free(derivatives); \
    derivatives = NULL;

    for (int inp_num = 0; inp_num < instance->input_n; inp_num++) {
        Tensor *current_inp = &instance->inputs[inp_num];
        Tensor *current_output = NULL;

        for (int layeri = 0; layeri < src_machine.layer_count; layeri++) {
            MLErr error = src_machine.layers[layeri]->forward(src_machine.layers[layeri], *current_inp, &current_output);
            if (error != ML_NO_ERR) {

                freeActivationsDerivatives;
                
                freeTensor(current_output);
                current_output = NULL;
                return error;
            }

            activations[inp_num][layeri] = *current_inp;
            current_inp = current_output;
        }

        freeTensor(current_output);

        // The value in current_output is the final output of the machine.
        // If the final value is needed outside of this context, the last activation should also be the last value, since:
        // The final layer is the error calculation.
        // Its syntax is described in the docs.

        // Calculate derivatives.
        Tensor *err_deriv = &instance->target_outputs[inp_num];
        Tensor *curr_deriv = err_deriv;
        Tensor *next_deriv = NULL;
        Tensor *self_deriv = NULL;

        for (int layeri = src_machine.layer_count - 1; layeri >= 0; layeri--) {
            MLErr error = src_machine.layers[layeri]->derive(src_machine.layers[layeri], *curr_deriv, activations[inp_num][layeri], &next_deriv, &self_deriv);
            if (curr_deriv != err_deriv) freeTensor(curr_deriv);
            if (error != ML_NO_ERR) {

                freeActivationsDerivatives;
                
                freeTensor(curr_deriv);
                curr_deriv = NULL;
                freeTensor(next_deriv);
                next_deriv = NULL;
                freeTensor(self_deriv);
                self_deriv = NULL;
                return error;
            }

            derivatives[inp_num][layeri] = *self_deriv;
            curr_deriv = next_deriv;
        } 
        // The final derivative is not needed, its the derivative
        // of the previous layer, but this is the last layer.
        freeTensor(next_deriv);

        // Run the optimizer. It is responsible for updating the weights.
        instance->optimizer(instance, activations[inp_num], derivatives[inp_num]);
    }

    freeActivationsDerivatives;

    return ML_NO_ERR;
}

MLErr mlSGD(LearningInstance *self, Tensor *activations, Tensor *derivatives) {
    // Multiply each derivative by the loss
    Tensor *learning_rate = matMakeTScalar(*(double *) self->hyper_parameters);
    Tensor *new_derivs = (Tensor *) malloc(sizeof(Tensor) * self->src_machine.layer_count);
    for (int i = 0; i < self->src_machine.layer_count; i++) {
        Tensor *new_deriv;
        matDot(derivatives[i], *learning_rate, &new_deriv);
        new_derivs[i] = *new_deriv;
    }

    MLErr error = ML_NO_ERR;
    // Update each layer with its derivative
    for (int i = 0; i < self->src_machine.layer_count; i++)  {
        error = self->src_machine.layers[i]->update(self->src_machine.layers[i], new_derivs[i]);
        if (error != ML_NO_ERR) {puts("errr"); goto Exit;}
    }

Exit:
    // Free new_deriv
    for (int i = 0; i < self->src_machine.layer_count; i++) freeTensorD(new_derivs[i]);
    free(new_derivs);

    return error;
}

MLErr mlSGDInitialize(LearningInstance *self) {
    if (self->hyper_parameters == NULL) return ML_NULL_PTR;
    self->_cache = NULL;

    return ML_NO_ERR;
}

MLErr mlSGDCleanup(LearningInstance *self) {
    return ML_NO_ERR;
}
