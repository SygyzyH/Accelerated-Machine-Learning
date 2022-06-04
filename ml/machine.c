#include "ml.h"
#include "../matrix/mat.h"

MLErr mlFeedForward(Machine machine, MatrixN inp, MatrixN **out) {
    if (out == NULL) return ML_NULL_PTR;

    Layer *l = machine.layers;
    MatrixN *rolling_output = NULL;
    MatrixN *last_output = &inp;

    while (l != NULL) {
        // Input of the next layer is the output of this layer.
        MLErr e = l->forawrd(l->parameters, *last_output, &rolling_output);
        if (last_output != &inp) freeMatrixN(last_output);
        last_output = rolling_output;
        
        // Propegate error.
        if (e != ML_NO_ERR) return e;

        l = l->next;
    }
    
    *out = rolling_output;

    return ML_NO_ERR;
}

MLErr mlAddLayer(Machine *machine, Layer *layer) {
    // If there's no first layer yet, set this layer as the first.
    if (machine->layers == NULL) {
        machine->layers = layer;
        return ML_NO_ERR;
    }

    Layer *current_layer = machine->layers;
    // Get to the end of the layer linked list.
    while (current_layer->next != NULL) current_layer = current_layer->next;
    // Link the new layer in.
    current_layer->next = layer;
    layer->prev = current_layer;

    return ML_NO_ERR;
}
