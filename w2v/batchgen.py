import collections
import random


class BatchGenerator:
    def __init__(self, data, skips_per_context, batch_size=32, transform_fun=lambda x: x):
        self.data = data
        self.datalen = len(data)
        self.skips_per_context = skips_per_context
        self.index = 0
        self.batch_size = batch_size
        self.transform_fun = transform_fun
        if (self.batch_size % skips_per_context) != 0:
            raise ValueError(
                "batch_size must be divisible with skips_per_context")
        self._init_context()

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size // self.skips_per_context):
            self._new_context()
            random.shuffle(self.index_list)
            indices = self.index_list[:self.skips_per_context]
            batch.extend([self.transform_fun(self.target)]
                         * self.skips_per_context)
            labels.extend([self.transform_fun(self.context[i])
                           for i in indices])

        return batch, labels


class ContinuousBatchGenerator(BatchGenerator):
    def __init__(self, data, skips_per_context, window, **kwargs):
        self.window = window
        super(ContinuousBatchGenerator, self).__init__(
            data, skips_per_context, **kwargs)
        if self.skips_per_context > 2*self.window:
            raise ValueError(
                "skips_per_context can't be larger than the context size")

    def _init_context(self):
        self.context = collections.deque(maxlen=2 * self.window + 1)
        self.context.extend(self.data[:2*self.window])
        self.index_list = list(range(self.window)) + \
            list(range(self.window+1, 2*self.window+1))
        self.index = 2*self.window-1

    def _new_context(self):
        self.index = (self.index + 1) % self.datalen
        self.context.append(self.data[self.index])
        self.target = self.context[self.window]
