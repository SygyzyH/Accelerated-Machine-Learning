#include "ml.h"

// NOTE: Random number generation uses time.
#include <time.h>
#include <math.h>

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

/* Fully Connected Layer */

MLErr mlFullyConnectedInitialize(Layer *self) {
    if (self->weights == NULL) return ML_LAYER_INVALID_WEIGHTS;
    if (((Tensor *) self->weights)->ndims != 2) return ML_LAYER_INVALID_WEIGHTS;

    return ML_NO_ERR;
}

MLErr mlFullyConnectedCleanup(Layer *self) {
    self->parameters = NULL;

    matFreeTensor((Tensor **) &self->weights);
    self->_cache = NULL;
    self->error = 0;

    return ML_NO_ERR;
}

MLErr mlFullyConnectedForward(Layer *self, Tensor *input, Tensor **output) {
    Tensor *weights = (Tensor *) self->weights;
    Tensor *inputf = matTensorFlatten(input, NULL);
    Tensor *res;

    MatrixErr e = matDot(weights, inputf, &res);

    switch (e) {
        case MAT_NO_ERROR: break;
        case MAT_DIMENSION_MISTMATCH:
            matFreeTensor(&inputf);
            matFreeTensor(&res);
            self->error = e;

            return ML_LAYER_INVALID_INPUT_DIMS;

        default:
            matFreeTensor(&inputf);
            matFreeTensor(&res);
            self->error = e;

            return ML_LAYER_INTERNAL_ERROR;
    }

    *output = res;

    matFreeTensor(&inputf);

    return ML_NO_ERR;
}

MLErr mlFullyConnectedDerive(Layer *self, Tensor *upstream_derivatives, Tensor *activation, Tensor **downstream_derivative, Tensor **self_derivative) {
    if (downstream_derivative == NULL) return ML_NULL_PTR;
    *downstream_derivative = NULL;
    if (self_derivative == NULL) return ML_NULL_PTR;
    *self_derivative = NULL;

    // self_derivative = activations * upstream_derivatives
    Tensor *activationf = matTensorFlatten(activation, NULL);
    // HACK: Is this correct? In GD it certainly is, but does it apply to any context?
    Tensor *upstream_derivativesf = matTensorFlatten(upstream_derivatives, NULL);
    Tensor *res;

    MatrixErr e = matDot(activationf, upstream_derivativesf, &res);

    switch (e) {
        case MAT_NO_ERROR: break;
        default:
            matFreeTensor(&activationf);
            matFreeTensor(&upstream_derivativesf);
            printf_m("%s\n", matGetErrorString(e));

            self->error = e;

            return ML_LAYER_INTERNAL_ERROR;
    }
    
    *self_derivative = res;

    matFreeTensor(&activationf);
    matFreeTensor(&upstream_derivativesf);

    // downstream_derivative = self->weights' * upstream_derivatives
    Tensor *weightsnT;
    matTTensor((Tensor *) self->weights, &weightsnT);

    Tensor *downstream;
    
    e = matDot(weightsnT, upstream_derivatives, &downstream);

    switch (e) {
        case MAT_NO_ERROR: break;
        default:
            matTensorPrint(weightsnT);
            matFreeTensor(&weightsnT);

            self->error = e;

            return ML_LAYER_INTERNAL_ERROR;
    }
    
    *downstream_derivative = downstream;

    matFreeTensor(&weightsnT);

    return ML_NO_ERR;
}

MLErr mlFullyConnectedUpdate(Layer *self, Tensor *self_derivative) {
    // self->weights -= self_derivative
    
    Tensor *new_weights;
    Tensor *weights = (Tensor *) self->weights;
    
    matSub(weights, self_derivative, &new_weights);
    
    self->weights = (void *) new_weights;

    return ML_NO_ERR;
}

const char* mlFullyConnectedErrorString(int error) {
    return "ML_LAYER_FULLY_CONNECTED_UNKNOWN_ERROR";
}

/* Bias */

MLErr mlBiasInitialize(Layer *self) {
    if (self->weights == NULL) return ML_LAYER_INVALID_PARAMETERS;
    return ML_NO_ERR;
}

MLErr mlBiasCleanup(Layer *self) {
    return ML_NO_ERR;
}

MLErr mlBiasForward(Layer *self, Tensor *input, Tensor **output) {
    Tensor *w = (Tensor *) self->weights;
    Tensor *it = matTensorFlatten(input, NULL);

    MatrixErr error = matAdd(w, it, output);
    matFreeTensor(&it);
    if (error != MAT_NO_ERROR) {
        self->error = error;

        return ML_LAYER_INTERNAL_ERROR;
    }

    return ML_NO_ERR;
}

MLErr mlBiasDerive(Layer *self, Tensor *upstream_derivatives, Tensor *activation, Tensor **downstream_derivative, Tensor **self_derivative) {
    // self_derivative = sum(upstream_derivatives)
    // downstream_derivative = upstream_derivatives
    
    *downstream_derivative = matTensorDeepCopy(upstream_derivatives, NULL);

    Tensor *w = (Tensor *) self->weights;
    Tensor *ones = mlWeightInitializer(ML_WEIGHT_INITIALIZER_ONES, w->ndims, w->dimsz);
    
    MatrixErr error = matDot(ones, upstream_derivatives, self_derivative);

    matFreeTensor(&ones);
    
    if (error != MAT_NO_ERROR) {
        self->error = error;

        return ML_LAYER_INTERNAL_ERROR;
    }

    return ML_NO_ERR;
}

MLErr mlBiasUpdate(Layer *self, Tensor *self_derivative) {
    // self->weights -= self_derivative
    Tensor *new_weights;
    Tensor *weightsn = (Tensor *) self->weights;

    MatrixErr error = matSub(weightsn, self_derivative, &new_weights);
    if (error != MAT_NO_ERROR) {
        self->error = error;

        return ML_LAYER_INTERNAL_ERROR;
    }
    
    self->weights = (void *) new_weights;

    matFreeTensor(&weightsn);

    return ML_NO_ERR;
}

const char* mlBiasErrorString(int error) {
    return "ML_LAYER_BIAS_UNKNOWN_ERROR";
}

/* ReLu */

MLErr mlReLuInitialize(Layer *self) {
    return ML_NO_ERR;
}

MLErr mlReLuCleanup(Layer *self) {
    return ML_NO_ERR;
}

MLErr mlReLuForward(Layer *self, Tensor *input, Tensor **output) {
    return ML_NO_ERR;
}

MLErr mlReLuDerive(Layer *self, Tensor *upstream_derivatives, Tensor *activation, Tensor **downstream_derivative, Tensor **self_derivative) {
    return ML_NO_ERR;
}

MLErr mlReLuUpdate(Layer *self, Tensor *self_derivative) {
    return ML_NO_ERR;
}

const char* mlReLuErrorString(int error) {
    return "ML_LAYER_RELU_UNKNOWN_ERROR";
}

/* MeanSquaredError */

MLErr mlMeanSquaredErrorInitialize(Layer *self) {
    return ML_NO_ERR;
}

MLErr mlMeanSquaredErrorCleanup(Layer *self) {
    return ML_NO_ERR;
}

MLErr mlMeanSquaredErrorForward(Layer *self, Tensor *input, Tensor **output) {
    *output = matTensorDeepCopy(input, NULL);

    return ML_NO_ERR;
}

MLErr mlMeanSquaredErrorDerive(Layer *self, Tensor *upstream_derivatives, Tensor *activation, Tensor **downstream_derivative, Tensor **self_derivative) {
    // upstream_derivatives = desired output
    // activation = actual output
    // downstream_derivative = error_function_DERIVATIVE(actual output, desired output)

    MatrixErr error = matSub(activation, upstream_derivatives, downstream_derivative);
    if (error != MAT_NO_ERROR) {
        self->error = error;

        return ML_LAYER_INTERNAL_ERROR;
    }
    
    // self_derivative is never used.
    *self_derivative = NULL;

    return ML_NO_ERR;
}

MLErr mlMeanSquaredErrorUpdate(Layer *self, Tensor *self_derivative) {
    return ML_NO_ERR;
}

const char* mlMeanSquaredErrorErrorString(int error) {
    return "ML_LAYER_MEAN_SQUARED_ERROR_UNKNOWN_ERROR";
}
