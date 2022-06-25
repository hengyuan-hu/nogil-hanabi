# nogil-hanabi

To reproduce the import problem
```
git pull
git git submodule update --init --recursive
make
python src/thread_loop.py
```

`test_with_dummy_actor` works fine as it does not use a pybinded batcher + torch model for inference.

`test`, which use the batcher tool from `rla` lib and a pytorch model, crashes on multiple threaeds.
