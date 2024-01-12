# cmake_minimum_required(VERSION 3.18)

# set(cuda_kernel_files
#     util.cc.cu
#     cross_entropy.cu
#     cublas_wrappers.cu
#     cuda_util.cu
#     dropout_kernels.cu
#     embedding_kernels.cu
#     embKernels.cc.cu
#     # fused_adam_kernel.cu
#     general_kernels.cu
#     gptKernels.cc.cu
#     llama_kernels.cu
#     normalize_kernels.cu
#     softmax_kernels.cu
#     softmax_kernels_new.cu
#     transform_kernels.cu
#     transform_kernels_new.cu
#     crf.cu
#     transformerKernels.cc.cu)

# add_library(lightseq_kernels STATIC ${cuda_kernel_files})
# target_link_libraries(lightseq_kernels PUBLIC -lcublas)
cmake_minimum_required(VERSION 3.18)
project(MySoftmaxProject LANGUAGES CXX CUDA) # Specify CUDA here

# List only the necessary CUDA kernel files for softmax
set(cuda_kernel_files
    softmax_kernels.cu
    # Add any other CUDA files that softmax_kernels.cu depends on
)

# Create a static library for the softmax kernel
add_library(softmax_kernels STATIC ${cuda_kernel_files})

# Include the directory where the header files are located
target_include_directories(softmax_kernels PUBLIC ${CMAKE_SOURCE_DIR}/includes)

# Link necessary CUDA libraries, such as cuBLAS
target_link_libraries(softmax_kernels PUBLIC -lcublas)

# If there are other dependencies (like custom libraries or CUDA utilities),
# link them here as well.


