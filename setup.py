from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_conv',
    ext_modules=[
        CUDAExtension(
            'custom_conv',
            sources=['csrc/custom_conv.cpp', 'csrc/custom_conv_kernel.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
