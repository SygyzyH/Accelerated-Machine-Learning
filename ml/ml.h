/* date = June 24th 2022 17:33 pm */

#ifndef ML_H
#define ML_H

#include "../matrix/mat.h"
#include <stdlib.h>

typedef enum {
    ML_NO_ERR=0,
    ML_LAYER_INVALID_PARAMETERS,
    ML_LAYER_INVALID_WEIGHTS,
    ML_LAYER_INVALID_INPUT_DIMS,
    ML_LAYER_INTERNAL_ERROR,
    ML_MACHINE_UNINITIALIZED_LAYER,
    ML_OPTIMIZER_UNEXPECTED_DIMS,
    ML_MAT_ERROR,
    ML_NULL_PTR
} MLErr;

static const char* mlGetErrorString(MLErr error) {
    switch (error) {
        case ML_NO_ERR: return "ML_NO_ERR";
        case ML_LAYER_INVALID_PARAMETERS: return "ML_LAYER_INVALID_PARAMETERS";
        case ML_LAYER_INVALID_WEIGHTS: return "ML_LAYER_INVALID_WEIGHTS";
        case ML_LAYER_INVALID_INPUT_DIMS: return "ML_LAYER_INVALID_INPUT_DIMS";
        case ML_LAYER_INTERNAL_ERROR: return "ML_LAYER_INTERNAL_ERROR";
        case ML_MACHINE_UNINITIALIZED_LAYER: return "ML_MACHINE_UNINITIALIZED_LAYER";
        case ML_OPTIMIZER_UNEXPECTED_DIMS: return "ML_OPTIMIZER_UNEXPECTED_DIMS";
        case ML_NULL_PTR: return "ML_NULL_PTR";
        default: return "Unknown ML Error";
    }
}

/* Layer */
// TODO: Have a flag in place to check if a layer is an error level. On the machine side, ensure the last layer is always an error evaluation layer.

// NOTE: Implemented functions must be named according to the doc.
// mlLayerNameFunctionName (for example mlFullyConnectedLayerDerive)
typedef struct layer {
    // Parameters of the layer, handled by the implementation.
    void *parameters;
    // Weights are handled by the implementation. An initial weight
    // pointer is given, and the implementation is responsible for
    // freeing it.
    void *weights;
    
    MLErr (*forward)(struct layer *self, Tensor input, Tensor **output);
    MLErr (*derive)(struct layer *self, Tensor upstream_derivatives, Tensor activation, Tensor **downstream_derivative, Tensor **self_derivative);
    MLErr (*update)(struct layer *self, Tensor self_derivative);

    // Cache for the implemntation to use and refrance as needed.
    // Handled by the implementation only, and is considered opaque
    // to the user.
    void *_cache;

    // Initialize the layer's variables and cache.
    MLErr (*initialize)(struct layer *self);
    // Cleanup function to free all reasources used by the layer,
    // and set them to NULL. 
    MLErr (*cleanup)(struct layer *self);

    // Error used to identify the layer that caused the machine to
    // crash.
    // If the error is zero, the layer is considered to be OK. If
    // the error is non-zero, it may be any number represanting the
    // internal error. The layer must still return standard `MLErr`
    // error codes on function returns.
    int error;
    // The implementation is responsible to implement a decoder to
    // the error numbers.
    const char* (*errorString)(int error);

    // Opaque error to keep track of which error occured during
    // initialization.
    MLErr _initialization_error;
} Layer;

// Prototype layer by name.
#define ML_PROTOTYPE_LAYER(name) \
MLErr ml##name##Initialize(Layer *self); \
MLErr ml##name##Cleanup(Layer *self); \
MLErr ml##name##Forward(Layer *self, Tensor input, Tensor **output); \
MLErr ml##name##Derive(Layer *self, Tensor upstream_derivatives, Tensor activation, Tensor **downstream_derivative, Tensor **self_derivative); \
MLErr ml##name##Update(Layer *self, Tensor self_derivative); \
const char* ml##name##ErrorString(int error);

// Make layer by name.
#define mlMakeLayer(name, parameters, initial_weights) mlMakeLayerExplicit(ml##name##Forward, ml##name##Derive, ml##name##Update, ml##name##Initialize, ml##name##Cleanup, ml##name##ErrorString, parameters, initial_weights)
// Make layer by explicit function pointers.
static Layer* mlMakeLayerExplicit(MLErr (*forward)(struct layer *self, Tensor input, Tensor **output), 
                 MLErr (*derive)(struct layer *self, Tensor upstream_derivatives, Tensor activation, Tensor **downstream_derivative, Tensor **self_derivative),
                 MLErr (*update)(struct layer *self, Tensor self_derivative),
                 MLErr (*initialize)(struct layer *self),
                 MLErr (*cleanup)(struct layer *self),
                 const char* (*errorString)(int error),
                 void *parameters, void *initial_weights) {
    Layer *l = (Layer *) malloc(sizeof(Layer));
    l->forward = forward;
    l->derive = derive;
    l->update = update;
    l->_cache = NULL;
    l->initialize = initialize;
    l->cleanup = cleanup;

    l->parameters = parameters;
    l->weights = initial_weights;    

    l->error = 0;
    l->errorString = errorString;

    l->_initialization_error = l->initialize(l);

    return l;
}

static inline void freeLayer(Layer *l) {
    if (l == NULL) return;
    l->cleanup(l);
}

/* Machine */

typedef struct {
    int layer_count;
    Layer **layers;

    int _all_layers_initialized;
} Machine;

static Machine mlMakeMachine(int layer_count, Layer **layers) {
    Machine m;

    m.layer_count = layer_count;
    m.layers = layers;

    m._all_layers_initialized = 1;
    for (int i = 0; i < m.layer_count; i++) 
        if (m.layers[i]->_initialization_error != ML_NO_ERR)
            m._all_layers_initialized = 0;

    return m;
}

MLErr mlMachineFeedForward(Machine machine, Tensor *input, Tensor **output);

typedef enum {
    ML_WEIGHT_INITIALIZER_ZEROS,
    ML_WEIGHT_INITIALIZER_ONES,
    ML_WEIGHT_INITIALIZER_GLOROT
} MLWeightInitializerType;

Tensor* mlWeightInitializer(MLWeightInitializerType initializer, unsigned ndims, unsigned *dims);

/* LearningInstance*/

// NOTE: Implemented functions must be named according to the doc.
typedef struct learninginstance {
    Machine src_machine;

    // Hyper parameters to be used by the optimizer implementation.
    // Handled by the implementation.
    void *hyper_parameters;

    // Number of inputs.
    int input_n;
    // Array of inputs to the machine. There is no guarentee the
    // NOTE:`Machine` isn't guarenteed to iterate over them linearly
    // this depends on the optimizer implementation. (even a
    // stochastic optimizer may randomize its inputs).
    Tensor *inputs;
    // Array of target outputs.
    // NOTE: `target_outputs` indecies must be aligned with `inputs`
    // Meaning the target output of input 1 is `target_outputs[1]`
    Tensor *target_outputs;

    // Optimier is handled by the implementation. 
    MLErr (*optimizer)(struct learninginstance *self, Tensor *activations, Tensor *derivatives);

    // Cache for the implemntation to use and refrance as needed.
    // Handled by the implementation only, and is considered opaque
    // to the user.
    void *_cache;

    // Initialize the instance's variables and cache.
    MLErr (*initialize)(struct learninginstance *self);
    // Cleanup function to free all reasources used by the instnace,
    // and set them to NULL. 
    MLErr (*cleanup)(struct learninginstance *self);
} LearningInstance;

// Prototype optimizer by name.
#define ML_PROTOTYPE_OPTIMIZER(name) \
MLErr ml##name(LearningInstance *self, Tensor *activations, Tensor *derivatives); \
MLErr ml##name##Initialize(LearningInstance *self); \
MLErr ml##name##Cleanup(LearningInstance *self);

// Make instance by name
#define mlMakeLearningInstance(machine, hyper_parameters, input_n, inputs, target_outputs, optimizer_name) mlMakeLearningInstanceExplicit(machine, hyper_parameters, input_n, inputs, target_outputs, ml##optimizer_name, ml##optimizer_name##Initialize, ml##optimizer_name##Cleanup)
// Make instnace by explicit function pointers.
static LearningInstance* mlMakeLearningInstanceExplicit(Machine machine, void *hyper_parameters, int input_n, Tensor *inputs, Tensor *target_outputs, 
    MLErr (*optimizer)(struct learninginstance *self, Tensor *activations, Tensor *derivatives),
    MLErr (*initialize)(struct learninginstance *self),
    MLErr (*cleanup)(struct learninginstance *self)) {
    LearningInstance *instance = malloc(sizeof(LearningInstance));

    instance->optimizer = optimizer;

    instance->_cache = NULL;
    instance->initialize = initialize;
    instance->cleanup = cleanup;

    instance->src_machine = machine;

    instance->hyper_parameters = hyper_parameters;

    instance->input_n = input_n;
    instance->inputs = inputs;
    instance->target_outputs = target_outputs;

    instance->initialize(instance);
    
    return instance;
}

MLErr mlTrainInstance(LearningInstance *instnace);

/* Prototype */

ML_PROTOTYPE_LAYER(FullyConnected);
ML_PROTOTYPE_LAYER(Bias);

ML_PROTOTYPE_LAYER(ReLu);
ML_PROTOTYPE_LAYER(Sigmoid);
ML_PROTOTYPE_LAYER(Tanh);

ML_PROTOTYPE_LAYER(MeanSquaredError);

ML_PROTOTYPE_OPTIMIZER(SGD);

#endif
