cmake_minimum_required (VERSION 3.1)
project (libnvm LANGUAGES CUDA C CXX)

set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 11)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)
find_package (CUDA 8.0)

set (NVIDIA "" CACHE PATH "Path to Nvidia driver source")
set (KERNEL "/lib/modules/${CMAKE_SYSTEM_VERSION}/build" CACHE PATH "Path to kernel source")

set (nvidia_archs "70;80;90" CACHE STRING "NVIDIA compute architectures")
set (no_module false CACHE BOOL "Do not build kernel module")
set (no_cuda false CACHE BOOL "Do not build with CUDA support")
set (BUILD_SHARED_LIBS true CACHE BOOL "Build shared library")

# GPU architecture flags
set (GPU_ARCHS "" CACHE INTERNAL "" FORCE)
foreach (arch ${nvidia_archs})
    string (CONCAT GPU_ARCHS ${GPU_ARCHS} " -gencode arch=compute_${arch},code=sm_${arch}")
endforeach ()

# Compiler flags
set (convert_warnings implicit-function-declaration incompatible-pointer-types int-conversion return-type unused-function unused-variable)
set (flags "")
foreach (flag ${convert_warnings})
    string (APPEND flags " -Werror=${flag}")
endforeach()
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra ${flags}" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++11" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -std=c++11" CACHE INTERNAL "" FORCE)
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -D__CUDA__ -D__CUDACC__ -O3;-std=c++11;${GPU_ARCHS}")

# Find Nvidia driver headers (for nv-p2p.h)
if (CUDA_FOUND AND NOT no_cuda)
    file (GLOB_RECURSE driver_paths "/usr/src/nvidia-*/nv-p2p.h")
    string (REPLACE "nv-p2p.h" "" driver_paths "${driver_paths}")
    if (driver_paths)
        list (REVERSE driver_paths)
    endif ()
    find_path (driver_include NAMES "nv-p2p.h" PATHS "${NVIDIA}" ${driver_paths})
    set (driver_dir "${driver_include}")
    if (NOT EXISTS "${driver_dir}/Module.symvers")
        get_filename_component (driver_dir "${driver_dir}" DIRECTORY)
    endif ()
    if ("${driver_dir}" STREQUAL "")
        message(STATUS "No NVIDIA driver symbols found")
    else ()
        message(STATUS "Using NVIDIA driver found in ${driver_dir}")
    endif ()
endif ()

# Library source files
set (libnvm_root "${PROJECT_SOURCE_DIR}/src")
file (GLOB libnvm_source "${libnvm_root}/*.cpp")
file (GLOB libnvm_linux_source "${libnvm_root}/linux/*.cpp")

# Module source files
set (module_root "${PROJECT_SOURCE_DIR}/module")

# Output directories
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (module_output "${CMAKE_BINARY_DIR}/module")

# Include directories
include_directories ("${PROJECT_BINARY_DIR}/include"
                     "${PROJECT_SOURCE_DIR}/include"
                     "${PROJECT_SOURCE_DIR}/include/freestanding/include")

# Build libnvm
CUDA_ADD_LIBRARY (libnvm ${libnvm_source} ${libnvm_linux_source})
target_include_directories (libnvm PRIVATE "${libnvm_root}")
target_link_libraries (libnvm ${CUDA_LIBRARIES} Threads::Threads)

if (CUDA_FOUND AND NOT no_cuda)
    target_compile_definitions(libnvm PRIVATE _CUDA __CUDA__)
endif ()

set_target_properties (libnvm PROPERTIES OUTPUT_NAME "nvm")

install (FILES ${libnvm_include} DESTINATION "include")
install (TARGETS libnvm DESTINATION "lib")

# Kernel module
if (NOT no_module)
    if (CUDA_FOUND AND NOT no_cuda AND EXISTS "${driver_dir}/Module.symvers")
        set (module_ccflags "-D_CUDA -I${driver_include} -I${libnvm_root} -I${libnvm_root}/linux")
        set (module_symbols "${driver_dir}/Module.symvers")
        message(STATUS "Configuring kernel module with CUDA")
    else ()
        set (module_ccflags "-I${libnvm_root} -I${libnvm_root}/linux")
        message(STATUS "Configuring kernel module without CUDA")
    endif ()
    configure_file ("${module_root}/Makefile.in" "${module_output}/Makefile" @ONLY)
    add_custom_target (kernel_module "${CMAKE_MAKE_PROGRAM}" WORKING_DIRECTORY "${module_output}")
endif ()

# Macro for benchmarks (used by subdirectories)
macro (make_benchmark target binary_name files)
    if (CUDA_FOUND AND NOT no_cuda)
        cuda_add_executable (${target} EXCLUDE_FROM_ALL ${files} OPTIONS ${GPU_ARCHS} -D__CUDA__)
        target_compile_definitions (${target} PRIVATE __CUDA__)
        add_dependencies (${target} libnvm)
        target_link_libraries (${target} libnvm ${CUDA_LIBRARIES})
        set_target_properties (${target} PROPERTIES OUTPUT_NAME "nvm-${binary_name}")
        list (APPEND benchmark_targets "${target}")
        set (benchmark_targets "${benchmark_targets}" PARENT_SCOPE)
    endif ()
endmacro ()
