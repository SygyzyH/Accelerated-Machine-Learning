# Accelerated Machine Learning
## Requirements
- OpenCL
- CMake

To compile from source an OpenCL implementation is required (Im using NVIDIA's).

## Compiling and running
Linux:
`sudo ./build.sh`

Windows:
`build.bat`

The script would, using CMake, automaticly compile **and install** the library as both a shared and a static library.
The script would also generate an executable to build and test an example model, which will than be ran.

To use the library, simply
```c
#include <ml.h>
```
And while compiling, ensure the compiler uses the correct library 
Linux: `libaml.so` or `libaml_static.a`
Windows: `libaml.dll` or `libaml_static.a`
shared or static, respectivly.

## Usage
### API
This project is ment to be expanded upon whenever I want / need to use new or diffrent types of layers.
For this reason, there are many types of APIs used for both interacting with the GPU (Used in some of my other projects)
and an API used for Tensor math.
Lastly, there is an API for the model creation and usage.

More spacific docs for each API can be found in the `docs` folder.

### oclapi
OpenCL API is used to:
- Streamline the calls to OpenCL in a clearer and more consice way
- Ensure safety by returning any errors that may come up
- Handeling device memory managment
- Allowing for baked-in static kernels (using CMake configuration)

This abstraction layer does however abstracts a lot of the underlying OpenCL functionality,
removing access to things such as custom types (float3, float4, etc.).

### mat.h
Tensor math library. Heavily relient on the aformentioned oclapi, but accelerated greatly.
Used as the mathematical engine in the machine learning model.

### ml.h
Library code itself.
Intended to be the interface between the back-end mat.h and oclapi and the end user.
- Implements the layers, using a specified protocol, allowing for easy chaining of layers
- Provides error checking, and contains the error in its internal structure
- Allows interaction with the model, via training and simple forward feeding
- Provides some utilities, such as weight initializers.

In order to use this in a project, **only** `ml.h` is required, since it imports all other APIs.
If the project only requires the **math** portion of this project, import `mat.h`.
For the accelerated functionality, import `oclapi.h`.

## TODO
- [X] mat.h is currently HIGHLY unsafe, and not throughly tested.
- [X] Compile this into a shared library.
- [X] Compile this into a static library.
- [ ] Better comments.
- [X] mat.h docs.
- [X] ml.h docs.
- [ ] oclapi docs.
- [ ] Allow `Optimizer`s to manage flowing derivatives during the learning process.
- [ ] mat.h can have a "calibration" phase in its init, checking when certain Tensors are faster to calculate on the CPU.
- [ ] add numpy-like `view`, or, like in previus commits, have the `Tensor` class not have guarenteed contigues data.
