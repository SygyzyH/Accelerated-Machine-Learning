/*
Register functions by passing their source code, and name.

Call the function using its name and passing its parameters.

Automatic clean-up.

TODO: Maybe also safe calls to run_kernel? (type checking, variable count checks, NULL ptr)
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>

#include "oclapi.h"

// Internal structures to keep track of kernel registrations
// and kernel argument structure.
typedef struct {
    char rettype[RETTYPE_SIZE];
    int isptr;
    size_t asize;
    // Temporary variable used during execution
    int _dsize;
    int _islocal;
    int _flags;
    void *_host_data;
    cl_mem _device_data;
} _oclapi_Karg;

typedef struct _oclapi_klist {
    cl_kernel kernel;
    char *name;
    
    int argc;
    // Array of args
    _oclapi_Karg *argv;
    
    struct _oclapi_klist *next;
} _oclapi_Klist;

cl_platform_id cpPlatform;
cl_device_id device_id;
cl_context context;
cl_command_queue queue;

OCLAPIErr oclerr = OCLNO_ERR;
cl_int clerr = CL_SUCCESS;

bool oclinit = false;

_oclapi_Klist *kernels = NULL;

// Get the last error
/*
returns the last error and reset it
*/
OCLAPIErr claGetError() {
    OCLAPIErr e = oclerr;
    oclerr = OCLNO_ERR;
    return e;
}

// Get the last OpenCL error
/*
returns the last OpenCL error and reset it
*/
cl_int claGetExtendedError() {
    cl_int e = clerr;
    clerr = CL_SUCCESS;
    return e;
}

// Initialize OpenCL and prepare registery
/*
returns 0 on success
*/
OCLAPIErr clainit() {
    int err;
    
    if ((err = clGetPlatformIDs(1, &cpPlatform, NULL))) goto ExitErrorCL;
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
    // If no GPU was found, use the CPU instead. Cant see how this would ever
    // fail, since the docs state the CPU is the host device - meaning, if this
    // code is running, there must be a CPU to run it. 
    if (err == CL_DEVICE_NOT_FOUND) {
        puts("Failed to get a GPU device, running un-accelerated.");
        err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    } if (err) goto ExitErrorCL;
    
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (err) goto ExitErrorCL;
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err) goto ExitErrorCL;
    
    puts("Initialized OpenCL API successfuly.");
    
    oclinit = true;
    
    return OCLNO_ERR;
    
    ExitErrorOCL:
    oclerr = err;
    
    return oclerr;
    
    ExitErrorCL:
    oclerr = OCLINTERNAL_OPENCL_ERROR;
    clerr = err;
    
    return oclerr;
}

// Cleanup registery
OCLAPIErr clacln() {
/*returns 0 on success*/
    if (!oclinit) return OCLUNINITIALIZED;
    
    _oclapi_Klist *k = kernels;
    
    while (k != NULL) {
        // TODO: Programs will repeat if multiple kernels are added at once.
        // This will cause clReleaseProgram to return an error. This is not
        // problematic per say but not best practice and should be avoided.
        cl_program prog;
        clGetKernelInfo(k->kernel, CL_KERNEL_PROGRAM, sizeof(cl_program), &prog, NULL);
        clReleaseProgram(prog);
        clReleaseKernel(k->kernel);
        
        _oclapi_Klist *oldk = k;
        free(k->argv);
        k = k->next;
        free(oldk);
    }
    
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    puts("Cleanup successful.");
    
    return OCLNO_ERR;
}

// Register a kerenel so it can be run
/*
src - source code string
kerneln - number of kernels in source code
... - string names of the kernels in source code
returns 0 on success
*/
/*
Before a function is ran using the ocl api, it must first be
registered. The user needs to supply a source program and any
number of kernels in that program. All names must be unique.
*/
OCLAPIErr claRegisterFromSrc(const char **src, int kerneln, ...) {
    int err;
    if (!oclinit) { err = OCLUNINITIALIZED; goto ExitErrorOCL; };
    
    cl_program prog = clCreateProgramWithSource(context, 1, src, NULL, &err);
    if (err) goto ExitErrorCL;
    // Build with kernel arg info flag to retrive it later. This solution thankfully
    // allows for building programs and having access to some of the information form
    // the compilation process, allowing for this whole library to feasably exist.
    err = clBuildProgram(prog, 0, NULL, "-cl-kernel-arg-info", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        char buildlog[BUILD_LOG_SIZE];
        clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, (size_t) BUILD_LOG_SIZE, buildlog, NULL);
        printf("Build error: \n%s\n", buildlog);
    } if (err) goto ExitErrorCL;
    
    va_list valist;
    va_start(valist, kerneln);
    
    for (int kernel = 0; kernel < kerneln; kernel++) {
        char *name = va_arg(valist, char *);
        
        _oclapi_Klist *k = kernels;
        
        if (k == NULL) {
            kernels = (_oclapi_Klist *) malloc(sizeof(_oclapi_Klist));
            k = kernels;
            k->next = NULL;
        } else {
            while (k->next != NULL) {
                if (strcmp(k->name, name) == 0) { 
                    va_end(valist);
                    return OCLINVALID_NAME;
                }
                
                k = k->next;
            }
            
            k->next = (_oclapi_Klist *) malloc(sizeof(_oclapi_Klist));
            k->next->next = NULL;
            k = k->next;
        }
        
        k->kernel = clCreateKernel(prog, name, &err);
        if (err) goto ExitErrorCL;
        k->name = name;
        clGetKernelInfo(k->kernel, CL_KERNEL_NUM_ARGS, sizeof(int), &(k->argc), NULL);
        k->argv = (_oclapi_Karg *) malloc(sizeof(_oclapi_Karg) * k->argc);
        
        // Fill out arguments
        for (int argument = 0; argument < k->argc; argument++) {
            clGetKernelArgInfo(k->kernel, argument, CL_KERNEL_ARG_TYPE_NAME, RETTYPE_SIZE, k->argv[argument].rettype, NULL);
            k->argv[argument].isptr = strchr(k->argv[argument].rettype, '*') != 0;
            
            if (strstr(k->argv[argument].rettype, "char")) {
                k->argv[argument].asize = sizeof(char);
            } else if (strstr(k->argv[argument].rettype, "int")) {
                k->argv[argument].asize = sizeof(int);
            } else if (strstr(k->argv[argument].rettype, "float")) {
                k->argv[argument].asize = sizeof(float);
            } else if (strstr(k->argv[argument].rettype, "double")) {
                k->argv[argument].asize = sizeof(double);
            } else {
                va_end(valist);
                puts("INVALID ARG");
                err = OCLINVALID_ARG;
                goto ExitErrorOCL;
            }
            
            cl_kernel_arg_address_qualifier addressq = CL_KERNEL_ARG_ADDRESS_PRIVATE;
            clGetKernelArgInfo(k->kernel, argument, CL_KERNEL_ARG_ADDRESS_QUALIFIER, sizeof(addressq), &addressq, NULL);
            k->argv[argument]._islocal = addressq == CL_KERNEL_ARG_ADDRESS_LOCAL;
            
            // Initilize values reset
            k->argv[argument]._dsize = 0;
            k->argv[argument]._flags = 0;
            k->argv[argument]._host_data = NULL;
        }
    }
    
    va_end(valist);
    
    return OCLNO_ERR;
    
    ExitErrorOCL:
    oclerr = err;
    
    return oclerr;
    
    ExitErrorCL:
    oclerr = OCLINTERNAL_OPENCL_ERROR;
    clerr = err;
    
    return oclerr;
}

// Run registered kernel
/*
name - name of the kernel
wdim - number of dimensions to run the kerenl in the GPU (typically between 1 to 3, see OpenCL docs)
gsz - array of sizes for the global run size. Array length should be dim (see OpenCL docs) 
lsz - array of sizes for the local run size. Array length should be dim (see OpenCL docs)
... - parameters for the function. After every pointer there must follow: size of the pointer, operation flags
returns 0 on success
*/
OCLAPIErr claRunKernel(const char *name, int wdim, size_t *gsz, size_t *lsz, ...) {
    int err;
    
    if (!oclinit) { err = OCLUNINITIALIZED; goto ExitErrorOCL; }
    
    _oclapi_Klist *k = kernels;
    
    while (k != NULL) {
        if (strcmp(k->name, name) == 0) break;
        k = k->next;
    }
    
    if (k == NULL) { err = OCLINVALID_NAME; goto ExitErrorOCL; }
    
    va_list valist;
    va_start(valist, lsz);
    
    for (int i = 0; i < k->argc; i++) {
        void *data;
        size_t device_dsize = k->argv[i].asize;
        
        if (k->argv[i].isptr) {
            if (strstr(k->argv[i].rettype, "char")) 
                k->argv[i]._host_data = (char *) (va_arg(valist, int *));
            else if (strstr(k->argv[i].rettype, "int"))
                k->argv[i]._host_data = (int *) va_arg(valist, int *);
            else if (strstr(k->argv[i].rettype, "float"))
                k->argv[i]._host_data = (float *) (va_arg(valist, double *));
            else if (strstr(k->argv[i].rettype, "double"))
                k->argv[i]._host_data = (double *) va_arg(valist, double *);
            else {
                va_end(valist);
                // TODO: Error message
                puts("TODO: Error message INVALID ARG");
                err = OCLINVALID_ARG;
                goto ExitErrorOCL;
            }
            
            // Store the data size to know how much data to return
            int dsize = va_arg(valist, int);
            k->argv[i]._dsize = dsize;
            
            // Store the flags in-case this argument needs to be copied out
            int flags = va_arg(valist, int);
            k->argv[i]._flags = flags;
            
            cl_mem_flags clflags;
            switch ((flags & ~OCLCPY) & ~OCLOUT) {
                case OCLREAD:
                clflags = CL_MEM_READ_ONLY;
                break;
                
                case OCLWRITE:
                clflags = CL_MEM_WRITE_ONLY;
                break;
                
                case OCLREAD | OCLWRITE:
                clflags = CL_MEM_READ_WRITE;
                break;
                
                default:
                va_end(valist);
                err = OCLINVALID_ARG;
                goto ExitErrorOCL;
            }
            
            // This will be freed later
            // TODO: Maybe the void *host_ptr argument can be used to save the copy?
            k->argv[i]._device_data = clCreateBuffer(context, clflags, k->argv[i].asize * dsize, NULL, &err);
            if (err) goto ExitErrorCL;
            
            if (!k->argv[i]._islocal) {
                data = &(k->argv[i]._device_data);
                device_dsize = sizeof(cl_mem);
            } else {
                data = NULL;
                // This took a bit too much to figure out... OpenCL docs are not the clearest...
                // Moreover, NVIDIA GPUs seem to be very lax when it comes to out of bounds access
                device_dsize = k->argv[i]._dsize * k->argv[i].asize;
            }
            
            // Copy data from given pointer to buffer pointed to by data
            if (flags & OCLCPY)
                if ((err = clEnqueueWriteBuffer(queue, *(cl_mem *) data, CL_TRUE, 0, k->argv[i].asize * dsize, k->argv[i]._host_data, 0, NULL, NULL))) goto ExitErrorCL;
        } else {
            if (strstr(k->argv[i].rettype, "char")) 
                data = &(char) { va_arg(valist, int) };
            else if (strstr(k->argv[i].rettype, "int")) 
                data = &(int) { va_arg(valist, int) };
            else if (strstr(k->argv[i].rettype, "float"))
                data = &(float) { va_arg(valist, double) };
            else if (strstr(k->argv[i].rettype, "double")) 
                data = &(double) { va_arg(valist, double) };
            else {
                va_end(valist);
                // TODO: Error message
                puts("TODO: Error message INVALID ARG");
                return OCLINVALID_ARG;
            }
        }
        
        if ((err = clSetKernelArg(k->kernel, i, device_dsize, data))) goto ExitErrorCL;
    }
    
    va_end(valist);
    
    // Run the kernel
    if ((err = clEnqueueNDRangeKernel(queue, k->kernel, wdim, NULL, gsz, lsz, 0, NULL, NULL))) goto ExitErrorCL;
    clFinish(queue);
    
    // Copy out the data and free it
    for (int i = 0; i < k->argc; i++) {
        if (k->argv[i].isptr) {
            if (k->argv[i]._flags & OCLOUT) {
                if ((err = clEnqueueReadBuffer(queue, k->argv[i]._device_data, CL_TRUE, 0, k->argv[i].asize * k->argv[i]._dsize, k->argv[i]._host_data, 0, NULL, NULL))) goto ExitErrorCL;
            }
            
            if ((err = clReleaseMemObject(k->argv[i]._device_data))) goto ExitErrorCL;
        }
    }
    
    clFinish(queue);
    
    return OCLNO_ERR;
    
    ExitErrorOCL:
    oclerr = err;
    
    return oclerr;
    
    ExitErrorCL:
    oclerr = OCLINTERNAL_OPENCL_ERROR;
    clerr = err;
    
    return oclerr;
}
