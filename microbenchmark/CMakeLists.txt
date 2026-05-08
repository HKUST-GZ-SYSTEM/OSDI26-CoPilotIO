cmake_minimum_required(VERSION 3.1)
project(copilotio_microbenchmark LANGUAGES CUDA C CXX)

set(BAM_DIR "${CMAKE_SOURCE_DIR}/../bam")
set(COPILOT_DIR "${CMAKE_SOURCE_DIR}/../copilot-io")
set(GDRCOPY_DIR "${CMAKE_SOURCE_DIR}/../gdrcopy")

set(nvidia_archs "70;80;90" CACHE STRING "NVIDIA compute architectures")
set(GPU_ARCHS "")
foreach(arch ${nvidia_archs})
    string(CONCAT GPU_ARCHS ${GPU_ARCHS} " -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

find_package(CUDA 8.0 REQUIRED)
find_package(Threads REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -D__CUDA__ -D__CUDACC__ -O3;-std=c++11;-Xcompiler;-fno-forward-propagate;${GPU_ARCHS}")

macro(add_bam_bench target src)
    cuda_add_executable(${target} ${src} OPTIONS ${GPU_ARCHS} -D__CUDA__)
    target_compile_definitions(${target} PRIVATE __CUDA__)
    target_include_directories(${target} PRIVATE
        "${BAM_DIR}/include"
        "${BAM_DIR}/include/freestanding/include")
    target_link_directories(${target} PRIVATE "${BAM_DIR}/build/lib")
    target_link_libraries(${target} nvm ${CUDA_LIBRARIES} Threads::Threads)
    set_target_properties(${target} PROPERTIES OUTPUT_NAME "nvm-${target}")
endmacro()

macro(add_copilot_bench target src)
    cuda_add_executable(${target} ${src} OPTIONS ${GPU_ARCHS} -D__CUDA__)
    target_compile_definitions(${target} PRIVATE __CUDA__)
    target_include_directories(${target} PRIVATE
        "${COPILOT_DIR}/include"
        "${COPILOT_DIR}/include/freestanding/include"
        "${GDRCOPY_DIR}/include")
    target_link_directories(${target} PRIVATE "${COPILOT_DIR}/build/lib" "${GDRCOPY_DIR}/src")
    target_link_libraries(${target} nvm ${CUDA_LIBRARIES} Threads::Threads gdrapi)
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "nvm-${target}"
        BUILD_RPATH "${GDRCOPY_DIR}/src"
        INSTALL_RPATH "${GDRCOPY_DIR}/src")
endmacro()

add_bam_bench(bam-read-bw bam_read_bw.cu)
add_copilot_bench(copilotio-read-bw copilotio_read_bw.cu)
