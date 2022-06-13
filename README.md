# nogil-hanabi

To see that pybind version crashes

```
git clone --recursive git@github.com:hengyuan-hu/nogil-hanabi.git
cd nogil-hanabi
pip install torch
pip install numpy
source load_env.sh. # for devfair machines
make
python -m src.test_pybind_env_speed
```

The program may crash when using more than 1 thread.


To see that the cffi version does not scale linearly:

```
cd thirdparty/hanabi-learning-environment
pip install .
cd ../..
python -m src.test_env_speed
```
example output
```
nogil=True
1 thread 1485.80 steps/s
2 thread 2572.90 steps/s
4 thread 4037.19 steps/s
8 thread 5624.91 steps/s
16 thread 5346.96 steps/s
32 thread 5154.30 steps/s
```
