# Accelerated Machine Learning
## Requirements
- OpenCL
- CMake

To compile from source an OpenCL implementation is required (Im using the NVIDIA's).

## Compiling and running
Linux:
`./build.sh`

Windows:
`build.bat`

The preloaded main is going to make a simple, two layered model, run it on an input, train it once and run again.

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
- Implements the layer codes, using a specified protocol, allowing for easy chaining of layers
- Provides error checking, and contains the error in its internal structure
- Allows interaction with the model, via training and simple forward feeding
- Provides some utilities, such as weight initializers.

## TODO
- [ ] mat.h can have a "calibration" phase in its init, checking when certain Tensors are faster to calculate on the CPU.
- [X] mat.h is currently HIGHLY unsafe, and not throughly tested.
- [ ] Compile this into a library.
