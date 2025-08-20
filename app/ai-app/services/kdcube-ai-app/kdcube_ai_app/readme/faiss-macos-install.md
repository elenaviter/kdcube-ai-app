
## Build from source
```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss
```

```bash
brew install libomp
brew install swig
brew install gflags
```

Check version of `gflags` with. In this example it is `2.2.2` - see usage below in config command.
```
brew info gflags
```

Configuration (note, this will install the lib in the location given in `CMAKE_INSTALL_PREFIX`)
```bash
  cmake -B build \
  -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_TESTS=OFF \
  -DFAISS_ENABLE_PERF_TESTS=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DOpenMP_C_FLAGS="-I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib" \
  -DOpenMP_CXX_FLAGS="-I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -Dgflags_DIR="/opt/homebrew/Cellar/gflags/2.2.2/lib/cmake/gflags" \
  -DCMAKE_INSTALL_PREFIX="$HOME/workdir/products/faiss_install" \
  .
```

Build
```bash
cmake --build build --target install
```

Add location of the lib in `LD_LIBRARY_PATH` (In `.zshrc`):
```bash
export DYLD_LIBRARY_PATH="$HOME/workdir/products/faiss_install/lib:$DYLD_LIBRARY_PATH"
```

In the shell where needed
```bash
source ~/.zshrc
```

## Install Python bindings
1. Activate my env
2. Install bindings (considering we are in `faiss` repo root).
```bash
cd build/faiss/python
python setup.py install
```
