# KnnGL

## Author

- **Name**: Deep Majumder
- **Roll Number**: 19CS30015

## Build

You need to have OpenGL libraries and headers installed. Additionally, you'd need a working installation of Python. After this, to build the Python library:

```sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

This will create `knngl.cpython-<major><minor>-<arch>-<os>-<abi>.so`. So on amd64 Linux with Python 3.11, this would be `knngl.cpython-311-x86_64-linux-gnu.so`. This shared library can be placed in any directory and we can simply use `knngl` by importing `import knngl`.

## Running

In order to run `knngl`, we need a "fake" screen to setup the OpenGL context. This can be done using `Xvfb` for XOrg.

```sh
Xvfb :99 -screen 0 1920x1080x24
```

Thereafter, you can set the `DISPLAY` environment variable to `:99` to set the fake screen.

```sh
DISPLAY=:99 python <file>.py
```

## Tests

Relevant tests are copied to the `build/` directory when invoking `cmake`. They must be run from the `build/` directory itself to ensure Python finds the `knngl` library. There are two tests:

- `test.py`: This is a trivial test and checks that KnnGL and Scikit return the same neighbours array.
- `test_adult.py`: This test runs KNN on the Adult dataset using both KnnGL and Scikit and prints out the classification accuracy.

## Benchmarks

Relevant benchmarks are copied to the `build/` directory when invoking `cmake`. They must be run from the `build/` directory itself to ensure Python finds the `knngl` library. There are two benchmarks:

- `bench_adult.py`: Runs the benchmark on Adult dataset and prints time taken for both KnnGL and Scikit.
- `bench_adult_rpi.py`: Same benchmarks as above but trimmed to run on the Raspberry Pi.

