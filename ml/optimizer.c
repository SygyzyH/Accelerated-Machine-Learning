#include "ml.h"
#include <mat.h>
#include <stdio.h>
#include <stdlib.h>

MLErr mlTrainInstance(LearningInstance *instance) {
    if (!instance->src_machine._all_layers_initialized) return ML_MACHINE_UNINITIALIZED_LAYER;

    // Convinience
    Machine src_machine = instance->src_machine;

    // First, collect the activations.
    // activations is an `input_n x layer_count` array of activations.
    // NOTE: `activations[0] = input`
    Tensor ***activations = (Tensor ***) malloc(sizeof(Tensor **) * instance->input_n);
    Tensor ***derivatives = (Tensor ***) malloc(sizeof(Tensor **) * instance->input_n);
    for (int i = 0; i < instance->input_n; i++) {
        activations[i] = (Tensor **) malloc(sizeof(Tensor *) * src_machine.layer_count);
        derivatives[i] = (Tensor **) malloc(sizeof(Tensor *) * src_machine.layer_count);
    }
    
    // Make sure that when data is freed on panic, itll free NULL instead of junk.
    for (int i = 0; i < instance->input_n; i++) 
        for (int j = 0; j < src_machine.layer_count; j++) {
            activations[i][j] = NULL;
            derivatives[i][j] = NULL;
        }

    // Clearer free of activations and derivatives. 
#define freeActivationsDerivatives \
    for (int i = 0; i < instance->input_n; i++) { \
        for (int j = 0; j < src_machine.layer_count; j++) { \
            if (j != 0) matFreeTensor(&activations[i][j]); \
            matFreeTensor(&derivatives[i][j]); \
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
            MLErr error = src_machine.layers[layeri]->forward(src_machine.layers[layeri], current_inp, &current_output);
            if (error != ML_NO_ERR) {

                freeActivationsDerivatives;
                
                matFreeTensor(&current_output);
                return error;
            }

            activations[inp_num][layeri] = current_inp;
            current_inp = current_output;
        }
        
        // Check if the output is the same shape as the expected output
        if (current_output->ndims != instance->target_outputs[inp_num].ndims) {

            freeActivationsDerivatives;

            matFreeTensor(&current_output);
            
            return ML_OPTIMIZER_UNEXPECTED_DIMS;
        }

        for (int dim = 0; dim < current_output->ndims; dim++)
            if (current_output->dimsz[dim] != instance->target_outputs[inp_num].dimsz[dim]) {
                
                freeActivationsDerivatives;

                matFreeTensor(&current_output);
                
                return ML_OPTIMIZER_UNEXPECTED_DIMS;
            }

        matFreeTensor(&current_output);

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
            MLErr derror = src_machine.layers[layeri]->derive(src_machine.layers[layeri], curr_deriv, activations[inp_num][layeri], &next_deriv, &self_deriv);
            if (curr_deriv != err_deriv) matFreeTensor(&curr_deriv);
            
            if (derror != ML_NO_ERR) {

                freeActivationsDerivatives;
                
                matFreeTensor(&curr_deriv);
                matFreeTensor(&next_deriv);
                matFreeTensor(&self_deriv);

                return derror;
            }

            derivatives[inp_num][layeri] = self_deriv;
            
            MLErr oerror = instance->propagate(instance, next_deriv, &curr_deriv);
            matFreeTensor(&next_deriv);
            
            if (oerror != ML_NO_ERR) {

                freeActivationsDerivatives;
                
                matFreeTensor(&curr_deriv);
                matFreeTensor(&next_deriv);
                matFreeTensor(&self_deriv);

                return oerror;
            }
        } 
        // The final derivative is not needed, its the derivative
        // of the previous layer, but this is the last layer.
        matFreeTensor(&next_deriv);

        // Run the optimizer. It is responsible for updating the weights.
        instance->optimizer(instance, activations[inp_num], derivatives[inp_num]);
    }

    freeActivationsDerivatives;

    return ML_NO_ERR;
}

MLErr mlSGD(LearningInstance *self, Tensor **activations, Tensor **derivatives) {
    Tensor *learning_rate = matMakeScalar(*(double *) self->hyper_parameters, NULL);
    int error = ML_NO_ERR;
    
    // Update each layer with its derivative
    for (int i = 0; i < self->src_machine.layer_count; i++)  {
        // Multiply each derivative by the learning rate
        Tensor *new_deriv = NULL;
        error = matDot(learning_rate, derivatives[i], &new_deriv);
        if (error != MAT_NO_ERROR) return ML_OPTIMIZER_INTERNAL_ERORR;

        error = self->src_machine.layers[i]->update(self->src_machine.layers[i], new_deriv);
        matFreeTensor(&new_deriv);
        if (error != ML_NO_ERR) return error;
    }

    return error;
}

MLErr mlSGDPropagate(LearningInstance *self, Tensor *upstream_derivative, Tensor **downstream_derivative) {
    *downstream_derivative = matTensorDeepCopy(upstream_derivative, NULL);

    return ML_NO_ERR;
}

MLErr mlSGDInitialize(LearningInstance *self) {
    if (self->hyper_parameters == NULL) return ML_NULL_PTR;
    self->_cache = NULL;

    return ML_NO_ERR;
}

MLErr mlSGDCleanup(LearningInstance *self) {
    return ML_NO_ERR;
}
