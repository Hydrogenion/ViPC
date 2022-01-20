from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='point_gpu',
    ext_modules=[
        CUDAExtension('point_gpu',
                      ['src/api.cpp',
                       'src/fps_wrapper.cpp',
                       'src/fps_cuda_impl.cu',
                       'src/ball_query_wrapper.cpp',
                       'src/ball_query_cuda_impl.cu'],
                      extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
