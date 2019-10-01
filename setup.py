from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

about = {}
with open('torchlars/__version__.py') as f:
    exec(f.read(), about)
version = about['__version__']
del about


with open('README.md') as f:
    long_description = f.read()

setup(
    name='torchlars',
    version=version,
    author='Kakao Brain',
    ext_modules=[
        CUDAExtension('torchlars._adaptive_lr', [
            'torchlars/adaptive_lr.cc',
            'torchlars/adaptive_lr_cuda.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension},
    description='A LARS implementation in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    maintainer='Chunmyong Park',
    zip_safe=False,
    packages=['torchlars'],
    install_requires=['torch'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
