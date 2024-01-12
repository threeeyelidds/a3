# cmake_minimum_required(VERSION 3.18)

# set(LS_PYBIND_KERNEL_FILES pybind_model.cpp)
# pybind11_add_module(lightseq MODULE ${LS_PYBIND_KERNEL_FILES})
# target_link_libraries(lightseq PUBLIC liblightseq lightseq_kernels)
# set_target_properties(lightseq PROPERTIES OUTPUT_NAME inference)
cmake_minimum_required(VERSION 3.18)

# Set the file that contains the pybind11 bindings for the softmax kernel
set(SOFTMAX_PYBIND_FILE pybind_kernel_cuda.cpp) # Replace with your actual file name

# Add pybind11 module
find_package(pybind11 REQUIRED)
pybind11_add_module(softmax_pybind MODULE ${SOFTMAX_PYBIND_FILE})

# Link with the softmax kernel library and any other necessary libraries
target_link_libraries(softmax_pybind PRIVATE softmax_kernels) # Replace 'softmax_kernels' with the actual target name of your softmax kernel library

# Optional: If you have a separate library for CUDA utilities or other dependencies, link them as well
# target_link_libraries(softmax_pybind PRIVATE your_additional_library)

# Set output name of the Python module
set_target_properties(softmax_pybind PROPERTIES OUTPUT_NAME softmax_module) # Replace 'softmax_module' with your desired module name
