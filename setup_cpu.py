#!/usr/bin/env python
"""
Implements the MDLP discretization criterion from Usama Fayyad's paper
"Multi-Interval Discretization of Continuous-Valued Attributes for
Classification Learning."

Now includes both CPU and GPU implementations.
"""

from setuptools import Extension, find_packages, setup

if __name__ == '__main__':
    # see https://stackoverflow.com/a/42163080 for the approach to pushing
    # numpy and cython dependencies into extension building only
    try:
        # if Cython is available, we will rebuild from the pyx file directly
        from Cython.Distutils import build_ext
        print("Using cython to build the extension")
        sources = ['mdlp/_mdlp.pyx']
    except:
        # else we build from the cpp file included in the distribution
        print("Cannot find available cython installation. Using cpp to build the extension")
        from setuptools.command.build_ext import build_ext
        sources = ['mdlp/_mdlp.cpp']

    class CustomBuildExt(build_ext):
        """Custom build_ext class to defer numpy imports until needed.

        Overrides the run command for building an extension and adds in numpy
        include dirs to the extension build. Doing this at extension build time
        allows us to avoid requiring that numpy be pre-installed before
        executing this setup script.
        """

        def run(self):
            import numpy
            self.include_dirs.append(numpy.get_include())
            build_ext.run(self)

    cpp_ext = Extension(
        'mdlp._mdlp',
        sources=sources,
        libraries=[],
        include_dirs=[],
        language='c++',
    )

    setup(
        name='mdlp-discretization',
        version='0.4.0',  # Bumped version for GPU support
        description=__doc__,
        license='BSD 3 Clause',
        url='github.com/hlin117/mdlp-discretization',
        author='Henry Lin',
        author_email='hlin117@gmail.com',
        install_requires=[
            'numpy>=1.11.2',
            'scipy>=0.18.1',
            'scikit-learn>=0.18.1',
            'cupy>=12.0.0',  # Added CuPy dependency for GPU support
        ],
        setup_requires=[
            'numpy>=1.11.2',
        ],
        packages=['mdlp'],
        ext_modules=[cpp_ext],
        cmdclass={'build_ext': CustomBuildExt},
        python_requires='>=3.8',  # Added Python version requirement for CuPy compatibility
    )
