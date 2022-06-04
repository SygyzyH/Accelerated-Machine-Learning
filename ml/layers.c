#include "ml.h"
#include <stdlib.h>

Layer* mlMakeLayer(MatrixN parameters, 
                  MLErr (*forawrd)(MatrixN parameters, MatrixN inp, MatrixN **out), 
                  MLErr (*getDelta)(MatrixN parameters, double prev_delta, double *new_delta)) {

    Layer *l = (Layer *) malloc(sizeof(Layer));

    l->parameters = parameters;
    l->forawrd = forawrd;
    l->getDelta = getDelta;
    l->next = NULL;
    l->prev = NULL;

    return l;
}

// Compute the forward output of a single fully connected layer.
/*
 * Only called by the ml library internally.
 * `parameters` - input parameters.
 * `inp` - input matrix.
 * `out` - pointer to layer's output.
 * returns error.
 * */
MLErr mlFullyConnectedLayerForward(MatrixN parameters, MatrixN inp, MatrixN **out) {
    if (out == NULL) return ML_NULL_PTR;
    *out = NULL;
    if (parameters.ndims != 2) return ML_LAYER_INVALID_PARAMETERS;

    Matrix2 *par_sub = matNAsMatrix2(parameters, parameters.dimsz[0], parameters.dimsz[1]);
    Matrix2 *inp_sub = matNAsMatrix2(inp, inp.literal_size, 1);
    Matrix2 *out_sub = NULL;
    if (par_sub == NULL || inp_sub == NULL) goto exitFailure;

    MatrixErr e = matMul(*inp_sub, *par_sub, &out_sub);
    
    if (e != MAT_NO_ERROR) goto exitFailure;

    *out = mat2AsMatrixN(*out_sub);
    
    freeMatrix2(out_sub);
    freeMatrix2(par_sub);
    freeMatrix2(inp_sub);

    return ML_NO_ERR;

exitFailure:
    freeMatrix2(par_sub);
    freeMatrix2(inp_sub);
    freeMatrix2(out_sub);

    return ML_LAYER_INVALID_PARAMETERS;
}

// TODO: This function.
MLErr mlFullyConnectedLayerGetDelta(MatrixN parameters, double prev_delta, double *new_delta) {
    
    return ML_NO_ERR;
}
