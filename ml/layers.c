#include "ml.h"

/* Fully Connected Layer */

MLErr mlFullyConnectedInitialize(Layer *self) {
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
    Tensor *downstream;
    
    e = matDot(*weightsnT, upstream_derivatives, &downstream);

    switch (e) {
        case MAT_NO_ERROR: break;
        default:
            freeTensor(weightsn);
            weightsn = NULL;
            freeTensor(weightsnT);
            weightsnT = NULL;

            self->error = e;

            return ML_LAYER_INTERNAL_ERROR;
    }
    
    *downstream_derivative = downstream;

    freeTensor(weightsn);
    weightsn = NULL;
    freeTensor(weightsnT);
    weightsnT = NULL;

    return ML_NO_ERR;
}

MLErr mlFullyConnectedUpdate(Layer *self, Tensor self_derivative) {
    // self->weights -= self_derivative
    // HACK: matPrintTensor(self_derivative);
    
    Tensor *new_weights;
    Tensor *weightsn = mat2AsTensor(*((Matrix2 *) self->weights));
    matSubT(*weightsn, self_derivative, &new_weights);
    
    Matrix2 *new_weights2 = matTAsMatrix2(*new_weights, weightsn->dimsz[0], weightsn->dimsz[1]);

    self->weights = (void *) new_weights2;

    freeTensor(weightsn);
    freeTensor(new_weights);

    // HACK: matPrintMatrix2(*(Matrix2 *) self->weights);

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

MLErr mlBiasForward(Layer *self, Tensor input, Tensor **output) {
    Tensor *w = (Tensor *) self->weights;

    MatrixErr error = matAddT(*w, input, output);
    if (error != MAT_NO_ERROR) {
        
        self->error = error;

        return ML_LAYER_INTERNAL_ERROR;
    }

    /* HACK: puts("====input");
    matPrintTensor(input);
    puts("====weights");
    matPrintTensor(*w);
    puts("====output");
    matPrintTensor(*out);*/

    return ML_NO_ERR;
}

MLErr mlBiasDerive(Layer *self, Tensor upstream_derivatives, Tensor activation, Tensor **downstream_derivative, Tensor **self_derivative) {
    // self_derivative = sum(upstream_derivatives)
    // downstream_derivative = upstream_derivatives
    *downstream_derivative = matTensorDeepCopy(upstream_derivatives);

    Tensor *w = (Tensor *) self->weights;
    Tensor *ones = mlWeightInitializer(ML_WEIGHT_INITIALIZER_ONES, w->ndims, w->dimsz);
    MatrixErr error = matDot(upstream_derivatives, *ones, self_derivative);

    freeTensor(ones);
    
    if (error != MAT_NO_ERROR) {
        
        self->error = error;

        return ML_LAYER_INTERNAL_ERROR;
    }

    matPrintTensor(**downstream_derivative);
    matPrintTensor(**self_derivative);

    return ML_NO_ERR;
}

MLErr mlBiasUpdate(Layer *self, Tensor self_derivative) {
    // self->weights -= self_derivative
    Tensor *new_weights;
    Tensor *weightsn = (Tensor *) self->weights;

    //matPrintTensor(self_derivative);
    //matPrintTensor(*weightsn);
    
    MatrixErr error = matSubT(*weightsn, self_derivative, &new_weights);
    if (error != MAT_NO_ERROR) {
        self->error = error;

        return ML_LAYER_INTERNAL_ERROR;
    }
    
    self->weights = (void *) new_weights;

    freeTensor(weightsn);

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

MLErr mlReLuForward(Layer *self, Tensor input, Tensor **output) {
    return ML_NO_ERR;
}

MLErr mlReLuDerive(Layer *self, Tensor upstream_derivatives, Tensor activation, Tensor **downstream_derivative, Tensor **self_derivative) {
    return ML_NO_ERR;
}

MLErr mlReLuUpdate(Layer *self, Tensor self_derivative) {
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

MLErr mlMeanSquaredErrorForward(Layer *self, Tensor input, Tensor **output) {
    *output = matTensorDeepCopy(input);

    return ML_NO_ERR;
}

MLErr mlMeanSquaredErrorDerive(Layer *self, Tensor upstream_derivatives, Tensor activation, Tensor **downstream_derivative, Tensor **self_derivative) {
    // upstream_derivatives = desired output
    // activation = actual output
    // downstream_derivative = error_function_DERIVATIVE(actual output, desired output)

    MatrixErr error = matSubT(activation, upstream_derivatives, downstream_derivative);
    if (error != MAT_NO_ERROR) {
        
        self->error = error;

        return ML_LAYER_INTERNAL_ERROR;
    }
    
    // self_derivative is never used.
    *self_derivative = mlWeightInitializer(ML_WEIGHT_INITIALIZER_ZEROS, 1, (int []) { 1 });

    return ML_NO_ERR;
}

MLErr mlMeanSquaredErrorUpdate(Layer *self, Tensor self_derivative) {
    return ML_NO_ERR;
}

const char* mlMeanSquaredErrorErrorString(int error) {
    return "ML_LAYER_MEAN_SQUARED_ERROR_UNKNOWN_ERROR";
}
