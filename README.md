# cuda_tests
CUDA exercises, e.g. from CUDA book (Programming Massively Parallel Processors 4th edition)

## Prerequisites

1. CUDA Toolkit installed
   ```bash
   # Check CUDA installation
   nvcc --version
   ```

2. Visual Studio Code with extensions:
   - Microsoft C/C++ Extension
   - CodeLLDB Extension

## Quick Start

1. Clone this repository
2. Open the folder in VSCode
3. Open `pmpp/ch01/vector_add.cu`
4. Set breakpoints by clicking left of line numbers
5. Press F5 to build and debug

## Building Without VSCode

```bash
cd pmpp/ch01
nvcc vector_add.cu -g -G -o vector_add
```

The `-g -G` flags enable debug information for both host and device code.
