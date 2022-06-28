#include "ml.h"

/* Fully Connected Layer */

MLErr mlFullyConnectedInitialize(Layer *self) {
    if (self->parameters != NULL) return ML_LAYER_INVALID_PARAMETERS;
    if (self->weights == NULL) return ML_LAYER_INVALID_WEIGHTS;

    return ML_NO_ERR;
}

MLErr mlFullyConnectedCleanup(Layer *self) {
    self->parameters = NULL;

    freeMatrix2((Matrix2 *) self->weights);
    self->weights = NULL;
    self->_cache = NULL;
    self->error = 0;

    return ML_NO_ERR;
}

MLErr mlFullyConnectedForward(Layer *self, Tensor input, Tensor **output) {
    Matrix2 *weights = (Matrix2 *) self->weights;
    Matrix2 *input2 = matTAsMatrix2(input, input.literal_size, 1);
    Matrix2 *res;

    MatrixErr e = matMul(*input2, *weights, &res);
    fprintf(stderr, "internal_error: %s\n", matGetErrorString(e));

    switch (e) {
        case MAT_NO_ERROR: break;
        case MAT_DIMENSION_MISTMATCH:
            freeMatrix2(input2);
            input2 = NULL;
            freeMatrix2(res);
            res = NULL;
            self->error = e;

            return ML_LAYER_INVALID_INPUT_DIMS;

        default:
            freeMatrix2(input2);
            input2 = NULL;
            freeMatrix2(res);
            res = NULL;
            self->error = e;

            return ML_LAYER_INTERNAL_ERROR;
    }

    *output = mat2AsTensor(*res);

    freeMatrix2(input2);
    input2 = NULL;
    freeMatrix2(res);
    res = NULL;

    return ML_NO_ERR;
}

MLErr mlFullyConnectedDerive(Layer *self, Tensor upstream_derivatives, Tensor activation, Tensor **downstream_derivative, Tensor **self_derivative) {
    if (downstream_derivative == NULL) return ML_NULL_PTR;
    *downstream_derivative = NULL;
    if (downstream_derivative == NULL) return ML_NULL_PTR;
    *self_derivative = NULL;

    matPrintTensor(upstream_derivatives);
    matPrintTensor(activation);

    // self_derivative = activations * upstream_derivatives
    Matrix2 *activation2 = matTAsMatrix2(activation, 1, activation.literal_size);
    // HACK: Is this correct? In GD it certainly is, but does it apply to any context?
    Matrix2 *upstream_derivatives2 = matTAsMatrix2(upstream_derivatives, upstream_derivatives.literal_size, 1);
    Matrix2 *res;

    MatrixErr e = matMul(*activation2, *upstream_derivatives2, &res);

    switch (e) {
        case MAT_NO_ERROR: break;
        default:
            freeMatrix2(activation2);
            activation2 = NULL;
            freeMatrix2(upstream_derivatives2);
            upstream_derivatives2 = NULL;

            self->error = e;

            return ML_LAYER_INTERNAL_ERROR;
    }
    
    *self_derivative = mat2AsTensor(*res);

    freeMatrix2(activation2);
    activation2 = NULL;
    freeMatrix2(upstream_derivatives2);
    upstream_derivatives2 = NULL;

    // downstream_derivative = self->weights * upstream_derivatives
    Tensor *weightsn = mat2AsTensor(*(Matrix2 *) self->weights);
    Tensor *weightsnT = matTTensor(*weightsn);
    
    *downstream_derivative = weightsnT;

    freeTensor(weightsn);
    weightsn = NULL;

    return ML_NO_ERR;
}

MLErr mlFullyConnectedUpdate(Layer *self, Tensor self_derivative) {
    // self->weights -= self_derivative
    if (self_derivative.ndims != 2) return ML_LAYER_INVALID_INPUT_DIMS;

    Matrix2 *new_weights;

    matSub(*((Matrix2 *) self->weights), *matTAsMatrix2(self_derivative, self_derivative.dimsz[0], self_derivative.dimsz[1]), &new_weights);
    /* HACK:
    puts("old:");
    matPrintMatrix2(*((Matrix2 *) self->weights));
    puts("new:");
    matPrintMatrix2(*new_weights);*/

    self->weights = (void *) new_weights;

    return ML_NO_ERR;
}

const char* mlFullyConnectedErrorString(int error) {
    return "ML_LAYER_FULLY_CONNECTED_UNKNOWN_ERROR";
}

/* MeanSquaredError */

MLErr mlMeanSquaredErrorInitialize(Layer *self) {
    return ML_NO_ERR;
}

MLErr mlMeanSquaredErrorCleanup(Layer *self) {
    return ML_NO_ERR;
}

MLErr mlMeanSquaredErrorForward(Layer *self, Tensor input, Tensor **output) {
    Tensor *input_cpy;

    input_cpy = matMakeTensor(input.ndims, input.dimsz);
    input_cpy->data = matTensorContiguousCopy(input);

    *output = input_cpy;

    return ML_NO_ERR;
}

MLErr mlMeanSquaredErrorDerive(Layer *self, Tensor upstream_derivatives, Tensor activation, Tensor **downstream_derivative, Tensor **self_derivative) {
    // upstream_derivatives = desired output
    // activation = actual output
    // downstream_derivative = error_function_DERIVATIVE(actual output, desired output)

    matSubTT(activation, upstream_derivatives, downstream_derivative);
    /* HACK:
    puts("activation:");
    matPrintTensor(activation);
    puts("upstream:");
    matPrintTensor(upstream_derivatives);
    puts("res:");
    matPrintTensor(**downstream_derivative);*/
    
    // self_derivative is never used.
    Tensor *self_derivative_stub;
    self_derivative_stub = matMakeTensor(1, (int []) { 1 });
    self_derivative_stub->data = malloc(sizeof(double) * self_derivative_stub->literal_size);
    self_derivative_stub->data[0] = -1;
    *self_derivative = self_derivative_stub;

    return ML_NO_ERR;
}

MLErr mlMeanSquaredErrorUpdate(Layer *self, Tensor self_derivative) {
    return ML_NO_ERR;
}

const char* mlMeanSquaredErrorErrorString(int error) {
    return "ML_LAYER_MEAN_SQUARED_ERROR_UNKNOWN_ERROR";
}
