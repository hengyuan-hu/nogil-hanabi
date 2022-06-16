# nogil-hanabi

To reproduce the import problem
```
git pull
git git submodule update --init --recursive
make
python -m src.test_import
```

It will crash when importing the second pybind library, no matter which one is the second.
