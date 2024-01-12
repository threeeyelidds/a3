from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup

setup(
    name='softmax_module',
    ext_modules=[
        CUDAExtension(
            name='softmax_module',
            sources=[
                './pybind/pybind_kernel_cuda.cpp',  # C++ Pybind definitions
                './cuda/softmax_kernels.cu',        # CUDA kernels
            ],
            include_dirs=['./cuda/includes'],      # Path to header files
            # If you have any libraries to link against:
            # libraries=['your_precompiled_library_name'],
            # library_dirs=['./cuda/lib'],         # Path to precompiled library files, if any
            # extra_compile_args={
            #     'cxx': ['-g'],
            #     'nvcc': ['-O2']
            # }
        )
    ],
    # cmdclass={
    #     'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    # }
)
