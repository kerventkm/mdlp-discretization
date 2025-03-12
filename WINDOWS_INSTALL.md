# Windows Installation Guide

This guide will help you set up the MDLP-discretization library with GPU support on Windows.

## Prerequisites

1. **Python Installation**
   - Download Python 3.8 or later from [Python.org](https://www.python.org/downloads/)
   - During installation:
     - ✅ Check "Add Python to PATH"
     - ✅ Check "Install pip"

2. **NVIDIA GPU Requirements**
   - NVIDIA GPU with CUDA support
   - Minimum GPU Driver version: 450.80.02
   - Check your GPU driver version:
     1. Right-click on desktop
     2. Select "NVIDIA Control Panel"
     3. Click "System Information" in bottom left
     4. Check "Driver Version"

3. **CUDA Toolkit Installation**
   1. Download CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
   2. Select:
      - Operating System: Windows
      - Architecture: x86_64
      - Version: Windows 10/11
      - Installer Type: exe (local)
   3. During installation:
      - Choose "Custom" installation
      - Make sure "CUDA" and "Development Runtime" are selected
      - Visual Studio integration is optional

## Installation Steps

1. **Create and activate a virtual environment** (recommended)
   ```bash
   # Open Command Prompt and navigate to your project directory
   cd your_project_directory

   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   venv\Scripts\activate
   ```

2. **Upgrade pip**
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Install dependencies**
   ```bash
   # Install all requirements
   pip install -r requirements.txt
   ```

4. **Install the MDLP library**
   ```bash
   pip install git+https://github.com/YOUR_USERNAME/mdlp-discretization
   ```

## Verification

Test if everything is installed correctly:

```python
import cupy as cp
import numpy as np
from mdlp.discretization_gpu import MDLP_GPU

# Create sample data
X = np.random.rand(1000, 4)
y = np.random.randint(0, 3, 1000)

# Initialize and run discretizer
discretizer = MDLP_GPU()
X_disc = discretizer.fit_transform(X, y)

print("Installation successful!")
```

## Troubleshooting

1. **CUDA version mismatch**
   - If you see CUDA version errors, modify requirements.txt to match your CUDA version:
     - For CUDA 11.x: use `cupy-cuda11x>=12.0.0`
     - For CUDA 10.2: use `cupy-cuda102>=12.0.0`

2. **GPU not found**
   - Verify GPU is recognized:
     ```python
     import cupy as cp
     print(cp.cuda.runtime.getDeviceCount())
     ```
   - Should return at least 1

3. **Import errors**
   - Make sure virtual environment is activated
   - Try reinstalling the package:
     ```bash
     pip uninstall mdlp-discretization
     pip install git+https://github.com/YOUR_USERNAME/mdlp-discretization
     ```

4. **Memory errors**
   - Reduce batch size or dataset size
   - Close other GPU-intensive applications

## Need Help?

If you encounter any issues:
1. Check NVIDIA driver is up to date
2. Verify CUDA Toolkit installation
3. Confirm Python version compatibility
4. Create an issue on the GitHub repository 