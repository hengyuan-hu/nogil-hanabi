import threading
import set_path
import torch
import rla
import utils


class BatchedModelWrapper:
    def __init__(self, model, verbose=False):
        self.model = model
        self._batchsizes = {}
        self._batchers = {}
        self._running_threads = []
        if verbose:
            self._actual_batchsizes = {}
        else:
            self._actual_batchsizes = None

    def register_method(self, method_name, batchsize):
        assert callable(getattr(self.model, method_name))
        assert method_name not in self._batchsizes
        assert method_name not in self._batchers

        self._batchsizes[method_name] = batchsize
        self._batchers[method_name] = rla.Batcher(batchsize)
        if self._actual_batchsizes is not None:
            self._actual_batchsizes[method_name] = []

    def async_call(self, method_name, data):
        return self._batchers[method_name].send(data)

    @utils.kill_all_on_failure
    def loop(self, method_name):
        batcher = self._batchers[method_name]
        while not batcher.terminated():
            batch = batcher.get()
            if len(batch) == 0:
                assert batcher.terminated()
                break

            if self._actual_batchsizes is not None:
                bsize = batch[next(iter(batch))].size(0)
                self._actual_batchsizes[method_name].append(bsize)

            result = getattr(self.model, method_name)(batch)
            batcher.set(result)
        return

    def start(self):
        assert len(self._running_threads) == 0
        for method_name in self._batchers:
            self._running_threads.append(
                threading.Thread(target=self.loop, args=(method_name,), daemon=True)
            )
            self._running_threads[-1].start()

    def stop(self):
        for _, batcher in self._batchers.items():
            batcher.exit()

        for thread in self._running_threads:
            thread.join()
