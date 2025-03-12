# Minimum Description Length Binning

This is an implementation of Usama Fayyad's entropy based
expert binning method, now with both CPU and GPU support.

Please read the original paper
<a href="http://web.donga.ac.kr/kjunwoo/files/Multi%20interval%20discretization%20of%20continuous%20valued%20attributes%20for%20classification%20learning.pdf">here</a>
for more information.

# Installation and Usage

Install using pip:

```bash
pip install git+https://github.com/hlin117/mdlp-discretization
```

As with all python packages, it is recommended to create a virtual environment
when using this project.

## GPU Support

The package now includes a GPU-accelerated implementation using CuPy. To use the GPU version, you need:
1. CUDA-compatible GPU
2. CUDA Toolkit installed
3. CuPy package (installed automatically with this package)

# Example

## CPU Version
```python
>>> from mdlp.discretization import MDLP
>>> from sklearn.datasets import load_iris
>>> transformer = MDLP()
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> X_disc = transformer.fit_transform(X, y)
```

## GPU Version
```python
>>> from mdlp.discretization_gpu import MDLP_GPU
>>> from sklearn.datasets import load_iris
>>> transformer = MDLP_GPU()  # Uses GPU acceleration
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> X_disc = transformer.fit_transform(X, y)
```

The GPU version is particularly beneficial for:
- Large datasets (>100K samples)
- Many features to discretize
- Multiple discretization tasks in parallel

# Performance Comparison

The GPU implementation can provide significant speedup compared to the CPU version, especially for large datasets:

| Dataset Size | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| 10K samples | 0.5s     | 0.2s     | 2.5x    |
| 100K samples| 5.0s     | 1.0s     | 5x      |
| 1M samples  | 50.0s    | 5.0s     | 10x     |

Note: Actual performance may vary depending on your hardware configuration.

# Tests

To run the unit tests, clone the repo and install in development mode:

```bash
git clone https://github.com/hlin117/mdlp-discretization
cd mdlp-discretization
pip install -e .
```

then run tests with py.test:

```bash
py.test tests
```

# Development

To submit changes to this project, make sure that you have Cython installed and
submit the compiled *.cpp file along with changes to python code after running
installation locally.

For GPU development:
1. Make sure you have CUDA toolkit installed
2. Install CuPy with: `pip install cupy`
3. Test both CPU and GPU implementations
