# ml.h API Protocol
The `ml.h` API is designed to be extendable, and, therefore, new layers and optimizers may be added,
either to the source code or through the library user.

Before usage, `mat.h` must be initialized.

## Layer
Prototype using 
```c
ML_PROTOTYPE_LAYER(name);
```

A layer is defined as
```c
typedef struct layer {
    // Parameters of the layer, handled by the implementation.
    void *parameters;
    // Weights are handled by the implementation. An initial weight
    // pointer is given, and the implementation is responsible for
    // freeing it.
    void *weights;
    
    MLErr (*forward)(struct layer *self, Tensor *input, Tensor **output);
    MLErr (*derive)(struct layer *self, Tensor *upstream_derivatives, Tensor *activation, Tensor **downstream_derivative, Tensor **self_derivative);
    MLErr (*update)(struct layer *self, Tensor *self_derivative);

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
} Layer;
```
> Note: Function names must follow the prototype convention `ml<LayerName><FunctionName>`

Each function gets the layer instance when called, and therefore has access to all fields.

When a function produces an error it should both return it and save it to the `error` field.

A `Layer` may implement its own `_cache`, `weights` and `parameters` structers, or expect any
arbitrary value to be passed to them.

A `Layer` is given an `initialize` function, to allow for any initilaization steps or parameter checks.
A `Layer` is also given a `cleanup` function, and is **responsible for freeing all data used by it**,
including `_chache`, `weights` and `parameters`.

A `Layer` must implement three processes used by the library:
- `forward` will recive both a pointer to the `Layer` instance and an input, and will produce the output.  
Input is **not** guarenteed to be a valid `Tensor`, or comply with the `Layer`'s specification.  
The output is expected to be a **valid** `Tensor`.
- `derive` will recive a pointer to the `Layer` instance, the `upstream_derivative` (i.e the derivative of the next layer),  
its own `activation`, and must return a **valid** `Tensor` to be passed down to the next `Layer`.  
`derive` may also produce its own `self_derivative`, which will be passed back to the `Layer` during the learning phase.
- `update` recives a pointer to the `Layer` instance, and its own `self_derivative`.  
The function is responsible for updating the `Layer` according to its own standard.

`Layer`s may be generated using
```c
mlMakeLayer(name, parameters, initial_weights)

Layer* mlMakeLayerExplicit(MLErr (*forward)(struct layer *self, Tensor *input, Tensor **output), 
                 MLErr (*derive)(struct layer *self, Tensor *upstream_derivatives, Tensor *activation, Tensor **downstream_derivative, Tensor **self_derivative),
                 MLErr (*update)(struct layer *self, Tensor *self_derivative),
                 MLErr (*initialize)(struct layer *self),
                 MLErr (*cleanup)(struct layer *self),
                 const char* (*errorString)(int error),
                 void *parameters, void *initial_weights);
```
To either make them explicitly or using their name only.
A `Layer` may fail to be created, in which case the resulting `Layer` is `NULL`.

Layers can be freed using
```c
void mlFreeLayer(Layer **l);
```
But should only be freed using their containg `Machine`.

Some utilities to generate initial weight values are also provided, namely
```c
Tensor* mlWeightInitializer(MLWeightInitializerType initializer, unsigned ndims, unsigned *dims);
```

Which accepts any of
```c
typedef enum {
    ML_WEIGHT_INITIALIZER_ZEROS,
    ML_WEIGHT_INITIALIZER_ONES,
    ML_WEIGHT_INITIALIZER_GLOROT
} MLWeightInitializerType;
```

As valid initializers.

## Machine
A container for multiple `Layers`.

Make using
```c
Machine mlMakeMachine(int layer_count, Layer **layers);
```
And free with 
```c
void mlFreeMachine(Machine **m);
```
To free a `Machine` refrance, and
```c
void mlFreeMachineD(Machine m);
```
To free a `Machine`.

A `Machine` may be used to iterate over a model using
```c
MLErr mlMachineFeedForward(Machine machine, Tensor *input, Tensor **output);
```

## LearningInstance
A `LearningInstance` is defined as
```c
typedef struct learninginstance {
    Machine src_machine;

    // Hyper parameters to be used by the optimizer implementation.
    // Handled by the implementation.
    void *hyper_parameters;

    // Number of inputs.
    int input_n;
    // Array of inputs to the machine. There is no guarentee the
    // NOTE: `Machine` isn't guarenteed to iterate over them linearly
    // this depends on the optimizer implementation. (even a
    // stochastic optimizer may randomize its inputs).
    Tensor *inputs;
    // Array of target outputs.
    // NOTE: `target_outputs` indecies must be aligned with `inputs`
    // Meaning the target output of input 1 is `target_outputs[1]`
    Tensor *target_outputs;

    // Optimier is handled by the implementation. 
    MLErr (*optimizer)(struct learninginstance *self, Tensor **activations, Tensor **derivatives);
    // Propagation handled by the implemntation.
    MLErr (*propagate)(struct learninginstance *self, Tensor *upstream_derivative, Tensor **downstream_derivative);

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
```

A `LearningInstance` has an `input_n` `inputs` and `target_outputs`.
While learning, the `LearningInstance` may iterate over them randomly, however,
`input` and `target_outputs` will always match.

A `LearningInstance` is made using
```c
mlMakeLearningInstance(machine, hyper_parameters, input_n, inputs, target_outputs, optimizer_name);

LearningInstance* mlMakeLearningInstanceExplicit(Machine machine, void *hyper_parameters, int input_n, Tensor *inputs, Tensor *target_outputs, 
    MLErr (*optimizer)(struct learninginstance *self, Tensor **activations, Tensor **derivatives),
    MLErr (*initialize)(struct learninginstance *self),
    MLErr (*cleanup)(struct learninginstance *self));
```

To either make them explicitly or using their name only.
A `LearningInstance` may fail to be created, in which case the resulting `LearningInstance` is `NULL`.

And can be freed with
> TODO

A `LearningInstance` must also use an `Optimizer` to update its weights.

### Optimizer
An `Optimizer` is prototyped using
```c
ML_PROTOTYPE_OPTIMIZER(name);
```

> Note: Function names must follow the prototype convention `ml<OptimizerName><FunctionName>`

An `Optimizer` is responsible for updating the weights, and it may apply transformations to their derivatives.
The `Optimizer` responsible of making sure it is able to do said transformations, or to error when they are not applicable to a `Layer`.

The `Optimizer` may expect spacific structres for its `hyper_parameters` and `_cache`.

An optimizer is given an `initialize` function to prepare its `_cache`, and validate its inputs and `hyper_parameters`. 
The same rules for a `Layer` initializer apply.
And an optimizer is given a `cleanup` function, and is responsible for freeing any resources it uses.

An `Optimizer` must implement two function:
- `optimizer` which will recive the `LearningInstance`, all the `activations` per `Layer` and all the `derivatives` per `Layer`,  
and will update each `Layer` in the `LearningInstance`s `Machine`.
- `propagate` will occour in every step of **back** propagation. This function is responsible for updateing the rolling weights,  
`upstream_derivative` and transforming it to `downstream_derivative`.

## Error
All `ml.h` functions that can produce errors (enumerated in `MLErr`) will either return them, or allow for a pointer to be passed
and filed with the coresponding error.

Any function that accepts a `MLErr` pointer may have `NULL` passed to them, which will ignore any error generated.

A function to convert enums of `MLErr` to `const char *` is provided
```c
const char* mlGetErrorString(MLErr error);
```
A comprehensive list of all errors
```c
typedef enum {
    ML_NO_ERR=0,                    // No error.
    ML_LAYER_INVALID_PARAMETERS,    // A layer has recived invalid parameters according to its specification.
    ML_LAYER_INVALID_WEIGHTS,       // A layer has recived invalid weights according to its specification.
    ML_LAYER_INVALID_INPUT_DIMS,    // A layer has recived invalid input dimensions according to its specification.
    ML_LAYER_INTERNAL_ERROR,        // A layer has encountered an unspecefied error, which can be retrived in its error field.
    ML_MACHINE_UNINITIALIZED_LAYER, // The machine in the instance has failed to initialize a layer.
    ML_OPTIMIZER_UNEXPECTED_DIMS,   // The optimizer has recived invalid input or target output dimensions.
    ML_OPTIMIZER_INTERNAL_ERORR,    // The optimizer has encountered an unspecefied error.
    ML_MAT_ERROR,                   // The function has encountered a mat.h library error which can be retrieved from the library.
    ML_NULL_PTR                     // The function has recived a NULL value as a parameter to a non-NULL input.
} MLErr;
```

> Note that `ML_NO_ERR` is guarenteed to be 0, and any other error is guarenteed to be non-zero.
