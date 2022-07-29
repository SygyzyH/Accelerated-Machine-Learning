# mat.h Library usage
## Application

The `mat.h` library must be initialized with
```c
MatrixErr matInit();
```

before usage, and will be cleaned up once `oclapi` is cleaned.

### Tensor struct
A `Tensor` is defined as

```c
typedef struct {
    unsigned *dimsz;
    unsigned ndims;

    size_t literal_size;

    double *data;
} Tensor;
```

Where `literal_size` > 0, and `data` is a contigues array.

One can construct a `Tesnor` with
```c
Tensor* matMakeTensor(unsigned ndims, unsigned *dims, MatrixErr *e);
```

A `Tensor` may also be a `scalar`. A rank-generic check for `scalar`s is
```c
int matIsTensorScalar(Tensor *t);
```

A `Tensor` may be defined as a scalar using
```c
Tensor* matMakeScalar(double s, MatrixErr *e);
```

`Tensors` may be accessed directly, using their `data` fileds, or using indecies and the supporting functions
```c
double* matTensorAtI(Tensor *t, unsigned *ind, MatrixErr *e);
unsigned *matTensorIAt(Tensor *t, int literal, MatrixErr *e);
```

### Operations
Many `Tensor` operations are defined in the `mat.h` library.
```c
MatrixErr matProd(Tensor *t1, Tensor *t2, Tensor **r); // Matrix product
MatrixErr matMult(Tensor *t1, Tensor *t2, Tensor **r); // Element-wise multiplication
MatrixErr matDot(Tensor *t1, Tensor *t2, Tensor **r);  // Dot product
MatrixErr matAdd(Tensor *t1, Tensor *t2, Tensor **r);  // Element-wise addition
MatrixErr matSub(Tensor *t1, Tensor *t2, Tensor **r);  // Element-wise subtraction

MatrixErr matTTensor(Tensor *t, Tensor **r);           // Tensor transpose (Shifting)
```

all element wise operations use fitting, using the fitting function
```c
MatrixErr matTensorFit(Tensor *t1, Tensor *t2, Tensor **t1r, Tensor **t2r);
```

Operations are not allowed to modify the source `Tensor`s once operation is completed,
and must return a new `Tensor` instances, or `NULL`.

### Utilities
Before using a `Tensor`, its validity should be ensured using
```c
MatrixErr matCheckTensor(Tensor *t, MatrixErr *e);
```

Some operations may access `Tensors` as if they were linear arrays, or vectors.
For that case, they can be "flattened"
```c
Tensor* matTensorFlatten(Tensor *t, MatrixErr *e);
```
And "reduced": (All dimensions equaling to `1` are removed)
```c
void matTensorReduce(Tensor *t);
```

`Tensors` can be printed using
```c
void matTensorPrint(Tensor *t);
```

and copied using
```c
Tensor* matTensorDeepCopy(Tensor *t, MatrixErr *e);
```

> Note: Deep copies will produce enteirly new copies of the `Tensor`, includeing data, which may be time-and-memory inefficiant.

Finally, `Tensors` can be freed using
```c
void matFreeTensor(Tensor **t);
```
Note: After free the pointer is set to NULL.
If this is not desiered, use
```c
void matFreeTensorD(Tensor t);
```

Which will only free the data and dimension fields.

## Errors
### MatrixErr
All `mat.h` functions that can produce errors (enumerated in `MatrixErr`) will either return them, or allow for a pointer to be passed
and filed with the coresponding error.

Any function that accepts a `MatrixErr` pointer may have `NULL` passed to them, which will ignore any error generated.

A function to convert enums of `MatrixErr` to `const char *` is provided
```c
const char* matGetErrorString(MatrixErr error);
```

A comprehensive list of all errors
```c
typedef enum {
    MAT_NO_ERROR=0,             // No error.
    MAT_INITIALIZATION_FAILED,  // On initilaiztion failure.
    MAT_UNINITIALIZED,          // When a function is called but initilaiztion failed.
    MAT_DIMENSION_MISTMATCH,    // A function recived Tensors who's dimensions don't match its specification.
    MAT_DIMENSION_OUT_OF_RANGE, // A function tried to access a Tensor at some index, who's out of the dimension's range.
    MAT_DIMENSION_ZERO,         // When a tensor has a zero sized dimension.
    MAT_KERNEL_FAILURE,         // An internal OCLAPI errored occoured.
    MAT_UNFIT_TENSORS,          // Tensors cannot be fit using fitting rules.
    MAT_TENSOR_NO_DATA,         // Tensor has invalid (NULL) data.
    MAT_TENSOR_NO_DIMS,         // Tensor has invalid (NULL) dimensions.
    MAT_NULL_PTR                // Recived a NULL pointer in place of a parameter that cannot be NULL.
} MatrixErr;
```
> Note that `MAT_NO_ERROR` is guarenteed to be 0, and any other error is guarenteed to be non-zero.

### OCLErr
If a problem has occoured in the kernel, the API will produce a `MAT_KERNEL_FAILURE` error.
To get it, access
```c
cl_int matGetExtendedError(int perserve);
```

Which will return the OpenCL error that occoured.
