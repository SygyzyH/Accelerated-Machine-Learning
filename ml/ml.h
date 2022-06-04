/* date = June 3rd 2022 15:01 pm */

#ifndef ML_H
#define ML_H

#include "../matrix/mat.h"

typedef enum {
    ML_NO_ERR,
    ML_LAYER_INVALID_PARAMETERS,
    ML_NULL_PTR
} MLErr;

typedef struct layer {
    MatrixN parameters;
    MLErr (*forawrd)(MatrixN parameters, MatrixN inp, MatrixN **out);
    MLErr (*getDelta)(MatrixN parameters, double prev_delta, double *new_delta); 

    struct layer *next;
    struct layer *prev;
} Layer;

typedef struct {
    Layer *layers;
    MatrixN hyper_parameters;
    
    MLErr (*learningAlgorithm)(MatrixN hyper_parameters, Layer *layers);
} Machine;

MLErr mlFeedForward(Machine machine, MatrixN inp, MatrixN **out);

MLErr mlAddLayer(Machine *machine, Layer *layer);
Layer* mlMakeLayer(MatrixN parameters, 
                  MLErr (*forawrd)(MatrixN parameters, MatrixN inp, MatrixN **out), 
                  MLErr (*getDelta)(MatrixN parameters, double prev_delta, double *new_delta));

MLErr mlFullyConnectedLayerForward(MatrixN parameters, MatrixN inp, MatrixN **out);
MLErr mlFullyConnectedLayerGetDelta(MatrixN parameters, double prev_delta, double *new_delta);

static const char* mlGetErrorString(MLErr error) {
    switch (error) {
        case ML_NO_ERR: return "ML_NO_ERR";
        case ML_LAYER_INVALID_PARAMETERS: return "ML_LAYER_INVALID_PARAMETERS";
        case ML_NULL_PTR: return "ML_NULL_PTR";
        default: return "Unknown ML error";
    }
}

#endif
