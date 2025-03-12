from setuptools import setup, find_packages

setup(
    name='mdlp-discretization',
    version='0.4.0',
    description='GPU-accelerated MDLP discretization',
    license='BSD 3 Clause',
    author='Henry Lin',
    author_email='hlin117@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.11.2',
        'scipy>=0.18.1',
        'scikit-learn>=0.18.1',
        'cupy-cuda12x>=12.0.0',  # For CUDA 12.x
    ],
    python_requires='>=3.8',
) 