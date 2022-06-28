cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
set(CMAKE_C_STANDARD 99)
# This header will be modified with CMake defines
set(CMAKE_MODIFIED_HEADER acceleration/kernels/static_kernels_src.h)
# Kernels are files located in KERNEL_SOURCE_DIRECTORY that are of type KERNEL_SOURCE_FILE_TYPE
set(KERNEL_SOURCE_DIRECTORY "acceleration/kernels/src")
set(KERNEL_SOURCE_FILE_TYPE "cl")

# TODO: Only do this if in debug mode
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

project(aml)

# For linux, if the OpenCL implementation is not found, using
# sudo apt install ocl-icd-opencl-dev
# may work (ubuntu).
# https://github.com/fireice-uk/xmr-stak-amd/issues/97
find_package(OpenCL REQUIRED)

add_executable(aml main.c acceleration/oclapi.c matrix/mat.c ml/layers.c ml/machine.c ml/optimizer.c)

# Put kernel sources into program build, staticly.
# Strings at the CMAKE_MODIFIED_HEADER_OUTPUT header constructed
# as "@KERNEL_STATIC_SOURCE_${KERNEL_SOURCE_FILE_TYPE_UPPER}@" will
# be replaced with the source code.
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "${KERNEL_SOURCE_DIRECTORY}/*.${KERNEL_SOURCE_FILE_TYPE}")
string(TOUPPER ${KERNEL_SOURCE_FILE_TYPE} KERNEL_SOURCE_FILE_TYPE_UPPER)

foreach(FILE_NAME ${KERNEL_SOURCES})
	set(BASE_NAME "KERNEL_STATIC_SOURCE_")
	# Get the file name. REGEX will look for some string between a and a .${KERNEL_SOURCE_FILE_TYPE}
	# For example, for "file/path/kernel_name.cl" the REGEX will return "kernel_name".
	string(REGEX MATCH "\/([^\/ ]+)\.${KERNEL_SOURCE_FILE_TYPE}$" THIS_FILE_NAME ${FILE_NAME})
	string(TOUPPER ${CMAKE_MATCH_1} THIS_FILE_NAME)
        # Append _CL
        string(APPEND BASE_NAME ${THIS_FILE_NAME} "_" ${KERNEL_SOURCE_FILE_TYPE_UPPER})
        file(STRINGS ${FILE_NAME} UNPREP_TEMP_STRING)
	# No option to not consume newlines and still keep them in parsed output
        list(JOIN UNPREP_TEMP_STRING "\\n" JOINT_TEMP_STRING)
	# Escape parentheses
        string(REPLACE "\"" "\\\"" JOINT_ESCAPED_TEMP_STRING "${JOINT_TEMP_STRING}")
        # Remove any opencl header inside kernel
        string(REGEX REPLACE "#include <opencl-c.h>|#include <opencl-c-base.h>" "" ${BASE_NAME} "${JOINT_ESCAPED_TEMP_STRING}")
        message("Defined source code \"@${BASE_NAME}@\".")
endforeach()

configure_file(${CMAKE_MODIFIED_HEADER} ${CMAKE_MODIFIED_HEADER} @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(aml OpenCL::OpenCL m)
