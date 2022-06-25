# nogil-hanabi

To reproduce the import problem
```
git pull
git git submodule update --init --recursive
make
python src/thread_loop.py
```

It will crash when importing the second pybind library, no matter which one is the second.
