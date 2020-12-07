
from torch.utils.data import Sampler, SequentialSampler, RandomSampler
from torch._six import int_classes as _int_classes
import numpy as np

class CustomSampler(Sampler):
    """Extending the Sampler to also pass a window size

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
            with ``__len__`` implemented.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        ws_range (List): The range in which the window size will be randomly sampled. 
        [low, high) -> low - inclusive, high - exclusive


    Example:
        >>> list(CustomSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False, ws_range=[16,32]))
        [[(0, 20), (1, 20), (2, 20)], [(3, 16), (4, 16), (5, 16)], [(6, 29), (7, 29), (8, 29)], [(9, 18)]]
    """

    def __init__(self, sampler, batch_size, drop_last=False, ws_range=[16,32]):
        # Validate input variables
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        assert len(ws_range) == 2

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.ws_range = ws_range

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                window_size = np.random.randint(*self.ws_range)
                batch = [(x, window_size) for x in batch]
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            window_size = np.random.randint(*self.ws_range)
            batch = [(x, window_size) for x in batch]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

if __name__ == "__main__":
    test = list(CustomSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False, ws_range=[16,32]))
    print(test)

    for batch in CustomSampler(RandomSampler(range(10)), batch_size=3, drop_last=False, ws_range=[16,32]):
        print(batch)