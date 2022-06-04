#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "../acceleration/oclapi.h"
#include "ml.h"

/* 
Internal ML functions, implemenation may vary in the future.
For the user end, use the function names to specify layer type 
when making a machine.
 */

bool mlinited = false;

// Initilize kernels for machine learning operations
/*
returns 0 on success
*/
int mlinit() {
    int err;
    
    char *mprog;
#define FILENAME ML_PROG
    mprog =
#include "../io/statichack.h"
    err = register_from_src(&(const char *) { mprog }, 5, 
                            "relu", "bnorm", "softmax", "maxpool", "conv2d");
    OCLAPI_FREE_IF_STATIC;
    safe(err);
    
    if (chkset(sets, DB))
        puts("Initialized ml successfuly.");
    
    mlinited = true;
    
    return MNO_ERR;
}

/*
Multidimensionality in memory is handled as:
for a tensor with 2nd degree dimensions of
w - 2
h - 2

and c extra dimensions, the final dimensions would be
c - 3
w - 2
h - 2 * c = 6

and the data would be laid out as such:

d1 d1
d1 d1
-----
d2 d2
d2 d2
-----
d3 d3
d3 d3
*/

int conv2d(double *params, int paramsw, int unused0,
           double *in, int inw, int inh, Mat **out) {
    /*
Parameters order:
- stridew
- strideh

- filterw
- filterh
- filtern

- pad
- channels

filtern * filterw * filterh * channels times:
- weights

filtern times:
- bias
*/
    
#pragma pack(push, 1)
    typedef struct {
        double stridew;
        double strideh;
        double filterw;
        double filterh;
        double filtern;
        double pad;
        double channels;
    } PARAMS;
#pragma pack(pop)
    
    if (paramsw < sizeof(PARAMS) / sizeof(double)) return MLINVALID_ARG;
    PARAMS *param = (PARAMS *) params;
    if (paramsw != sizeof(PARAMS) / sizeof(double) + (param->filterw * param->filterh * param->channels + 1) * param->filtern) return MLINVALID_ARG;
    
    // ooo spooky pointer math, im so scared let me call mommy java to pick me up
    double *weights = (double *) params + sizeof(PARAMS) / sizeof(double);
    int weightn = (int) param->filterw * param->filterh * param->channels * param->filtern;
    double *bias = weights + weightn;
    int biasn = (int) param->filtern;
    
    int einw = inw, einh = (int) inh / param->channels;
    
    int ptw, pth;
    int ptop, pbot, pleft, pright;
    switch ((int) param->pad) {
        case SAME: {
            // In same padding mode, the output size would be input size / stride.
            // Using the formula (input size - pool size + 2 * padding) / stride + 1 = output
            // Solving for 2 * padding gives padding = pool size - stride.
            // Than the padding of each dimension is padding / 2. Padding doesn't have to
            // be devisable by 2, so to solve that one side is given the reminder.
            // Matlab pads with the ceil on bottom and right side
            int ow = ceil((double) einw / param->stridew);
            int oh = ceil((double) einh / param->strideh);
            int pw = (ow - 1) * param->stridew + param->filterw;
            int ph = (oh - 1) * param->strideh + param->filterh;
            
            ptw = MAX(0, pw - einw);
            pth = MAX(0, ph - einh);
            
            ptop = floor((double) pth / 2);
            pbot = ceil((double) pth / 2);
            pleft = floor((double) ptw / 2);
            pright = ceil((double) ptw / 2);
            
            break;
        }
        
        case NONE: {
            ptw = 0;
            pth = 0;
            
            ptop = 0;
            pbot = 0;
            pright = 0;
            pleft = 0;
            
            break;
        }
        
        default:
        if (chkset(sets, DB)) puts("ML_C: Convelution layer FAILURE - Invalid padding type.");
        return MLINVALID_ARG;
    }
    
    Mat *o = (Mat *) malloc(sizeof(Mat));
    o->width = floor((einw - param->filterw + ptw) / param->stridew) + 1;
    o->height = floor(param->filtern * ((einh - param->filterh + pth) / param->strideh + 1));
    o->data = (double *) malloc(sizeof(double) * o->width * o->height);
    
    int ew = o->width, eh = (int) o->height / param->filtern;
    size_t gz[] = { param->filtern * param->channels, ew * param->filterw, eh * param->filterh };
    size_t lz[] = { param->channels, param->filterw, param->filterh };
    
    run_kernel("conv2d", 3, gz, lz,
               in, inw * inh, OCLREAD | OCLCPY,
               inw, inh, (int) param->stridew, (int) param->strideh,
               pleft, pright, ptop, pbot,
               NULL, (int) (param->filterw * param->filterh * param->channels), OCLREAD | OCLWRITE,
               bias, biasn, OCLREAD | OCLCPY,
               weights, weightn, OCLREAD | OCLCPY,
               o->data, o->width * o->height, OCLWRITE | OCLOUT);
    
    *out = o;
    
    return MLNO_ERR;
}

int maxpool(double *params, int paramsw, int unused0,
            double *in, int inw, int inh, Mat **out) {
    
#pragma pack(push, 1)
    typedef struct {
        double stridew;
        double strideh;
        double poolw;
        double poolh;
        double pad;
        double channels;
    } PARAMS;
#pragma pack(pop)
    
    if (paramsw != sizeof(PARAMS) / sizeof(double)) {
        if (chkset(sets, DB))
            puts("ML_C: Maxpool layer FAILURE - Parameter size mistmatch.");
        return MLINVALID_ARG;
    }
    
    PARAMS *param = (PARAMS *) params;
    
    int einw = inw, einh = (int) inh / param->channels;
    
    int ptw, pth;
    int ptop, pbot, pleft, pright;
    switch ((int) param->pad) {
        case SAME: {
            // In same padding mode, the output size would be input size / stride.
            // Using the formula (input size - pool size + 2 * padding) / stride + 1 = output
            // Solving for 2 * padding gives padding = pool size - stride.
            // Than the padding of each dimension is padding / 2. Padding doesn't have to
            // be devisable by 2, so to solve that one side is given the reminder.
            // Matlab pads with the ceil on bottom and right side
            int ow = ceil((double) einw / param->stridew);
            int oh = ceil((double) einh / param->strideh);
            int pw = (ow - 1) * param->stridew + param->poolw;
            int ph = (oh - 1) * param->strideh + param->poolh;
            
            ptw = MAX(0, pw - einw);
            pth = MAX(0, ph - einh);
            
            ptop = floor((double) pth / 2);
            pbot = ceil((double) pth / 2);
            pleft = floor((double) ptw / 2);
            pright = ceil((double) ptw / 2);
            
            break;
        }
        
        case NONE: {
            ptw = 0;
            pth = 0;
            
            ptop = 0;
            pbot = 0;
            pright = 0;
            pleft = 0;
            
            break;
        }
        
        default:
        if (chkset(sets, DB)) puts("ML_C: Maxpool layer FAILURE - Invalid padding type.");
        return MLINVALID_ARG;
    }
    
    Mat *o = (Mat *) malloc(sizeof(Mat));
    o->width = floor((einw - param->poolw + ptw) / param->stridew) + 1;
    o->height = floor(param->channels * ((einh - param->poolh + pth) / param->strideh + 1));
    o->data = (double *) malloc(sizeof(double) * o->width * o->height);
    
    int ew = o->width, eh = (int) o->height / param->channels;
    size_t gz[] = { ew * param->poolw, eh * param->poolh, param->channels };
    size_t lz[] = { param->poolw, param->poolh, 1 };
    
    run_kernel("maxpool", 3, gz, lz,
               in, inw * inh, OCLREAD | OCLCPY, inw, inh,
               (int) param->stridew, (int) param->strideh, pleft, pright, ptop, pbot,
               NULL, (int) (param->poolw * param->poolh), OCLREAD | OCLWRITE,
               o->data, o->width * o->height, OCLWRITE | OCLOUT);
    *out = o;
    
    return MLNO_ERR;
}

int fullyc(double *params, int paramsw, int unused0,
           double *in, int inw, int inh, Mat **out) {
    /*
Parameters order:
- in size
- out size

insz * outsz times:
 - weight

outsz times:
- bias
*/
    
#pragma pack(push, 1)
    typedef struct {
        double insz;
        double outsz;
    } PARAMS;
#pragma pack(pop)
    
    if (paramsw < sizeof(PARAMS) / sizeof(double)) return MLSIZE_MISMATCH;
    PARAMS *param = (PARAMS *) params;
    int insz = (int) param->insz, outsz = (int) param->outsz;
    if (paramsw != (insz + 1) * outsz + sizeof(PARAMS) / sizeof(double)) return MLSIZE_MISMATCH;
    
    if (inw * inh != insz) return MLINVALID_ARG;
    
    double *weights = (double *) params + sizeof(PARAMS) / sizeof(double);
    int weightn = insz * outsz;
    double *bias = (double *) weights + weightn;
    int biasn = outsz;
    
    Mat *o = (Mat *) malloc(sizeof(Mat));
    o->width = outsz;
    o->height = inw;
    
    double *m = NULL;
    matmul(in, inw, inh, weights, insz, outsz, &(m));
    matadd(m, o->width, o->height, bias, outsz, 1, &(o->data));
    
    free(m);
    
    *out = o;
    
    return MLNO_ERR;
}

int zcenter(double *params, int paramsw, int unused0,
            double *in, int inw, int inh, Mat **out) {
    /*
Parameters order:
insz times
- mean
*/
    
    int insz = inw * inh;
    if (paramsw != insz) return MLINVALID_ARG;
    
    Mat *o = (Mat *) malloc(sizeof(Mat));
    o->width = inw;
    o->height = inh;
    
    matsub(in, inw, inh, params, inw, inh, &(o->data));
    
    *out = o;
    
    return MLNO_ERR;
}

int bnorm(double *params, int paramsw, int unused0,
          double *in, int inw, int inh, Mat **out) {
    /*
Parameters order:
- channels

   insz times
- mean
- variance
- offset
- scale
*/
    
#pragma pack(push, 1)
    typedef struct {
        double mean;
        double variance;
        double offset;
        double scale;
    } PARAMS;
#pragma pack(pop)
    
    if (paramsw < 1) return MLSIZE_MISMATCH;
    int channels = params[0];
    
    int insz = inw * inh;
    if (paramsw - 1 != channels * sizeof(PARAMS) / sizeof(double)) return MLSIZE_MISMATCH;
    
    Mat *o = (Mat *) malloc(sizeof(Mat));
    o->width = inw;
    o->height = inh;
    o->data = (double *) malloc(sizeof(double) * o->width * o->height);
    
    int einsz = insz / channels;
    
    size_t gz[] = { einsz, channels };
    
    int e = run_kernel("bnorm", 2, gz, NULL,
                       in, insz, OCLREAD | OCLCPY,
                       einsz, channels,
                       params, paramsw, OCLREAD | OCLCPY,
                       o->data, insz, OCLWRITE | OCLOUT);
    
    *out = o;
    
    return MLNO_ERR;
}

int dropout(double *prob, int unused0, int unused1,
            double *in, int iw, int ih, Mat **out) {
    // Probability for dropout
    double p = *prob;
    
    // p must be in range 0 < p < 1
    if (1 < p || p < 0) return MLINVALID_ARG;
    
    Mat *o = (Mat *) malloc(sizeof(Mat));
    o->width = iw;
    o->height = ih;
    o->data = (double *) malloc(sizeof(double) * iw * ih);
    
    srand(time(NULL));
    for (int i = 0; i < ih; i++) {
        for (int j = 0; j < iw; j++) {
            if (((double) rand() / RAND_MAX) < p)
                o->data[j + i * iw] = 0;
            else
                o->data[j + i * iw] = in[j + i * iw];
        }
    }
    
    *out = o;
    
    return MLNO_ERR;
}

int softmax(double *unused0, int unused1, int unused2,
            double *in, int iw, int ih, Mat **out) {
    int insz = iw * ih;
    
    Mat *o = (Mat *) malloc(sizeof(Mat));
    o->width = insz;
    o->height = 1;
    o->data = (double *) malloc(sizeof(double) * insz);
    
    size_t gz[] = { insz };
    
    double *temp;
    run_kernel("softmax", 1, gz, NULL, 
               in, insz, OCLREAD | OCLCPY,
               NULL, insz, OCLREAD | OCLWRITE,
               o->data, insz, OCLWRITE | OCLOUT);
    
    *out = o;
    
    return MLNO_ERR;
}

int relu(double *unused0, int unused1, int unused2,
         double *in, int inw, int inh, Mat **out) {
    
    Mat *o = (Mat *) malloc(sizeof(Mat));
    o->width = inw;
    o->height = inh;
    o->data = (double *) malloc(sizeof(double) * inw * inh);
    
    int insz = inw * inh;
    
    size_t gz[] = { insz };
    
    run_kernel("relu", 1, gz, NULL, 
               in, insz, OCLREAD | OCLCPY,
               o->data, insz, OCLWRITE | OCLOUT);
    
    *out = o;
    
    return MLNO_ERR;
}

// Pass an input through a machine
/*
machine - machine to run the input on
input - input matrix
oputput - resulting output (memory allocated automaticly)
returns 0 on success
*/
int forwardpass(Layer machine, Mat input, Mat **output) {
    Layer *p = &machine;
    Mat *datai = &input;
    Mat *prevout = NULL;
    
    while (p != NULL) {
        // TODO: Does size checking need to happen before the layer is called? why not in the
        // layer implementation itself?
        
        // If there needs to be size checking
        if(p->inw != -1 && p->inh != -1)
            // Ensure the size is correct
            if (p->inw != datai->width || p->inh != datai->height) return MLSIZE_MISMATCH;
        
        // Run the data through the current layer
        if((*p->transform)(unpkmat(p->params), unpkmatp(datai), output)) return MLLAYER_ERR;
        
        // Move to the next layer
        datai = *output;
        p = p->next;
        
        if (prevout != NULL)
            free(prevout->data);
        // free-ing NULL is allowed
        free(prevout);
        
        prevout = *output;
    }
    
    return MLNO_ERR;
}

// Adds a layer to a machine
/*
machine - machine to add the layer to
layer - layer to be added
*/
void addlayer(Layer **machine, Layer **newl) {
    Layer *p = *machine;
    
    while (p->next != NULL)
        p = p->next;
    
    (*newl)->prev = p;
    p->next = *newl;
}
