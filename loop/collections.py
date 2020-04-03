from collections import OrderedDict
from typing import List, Union


class NamedList:
    """A convenience wrapper that allows to iterate over ordered dict values and
    access its elements using numerical indexes.
    """
    def __init__(self, od: Union[OrderedDict, List]):
        self.od = OrderedDict(od)
        self.names = list(self.od)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index >= len(self):
            raise StopIteration()
        return self[self.index]

    def __len__(self):
        return len(self.od)

    def __getitem__(self, item: Union[str, int]):
        if isinstance(item, str):
            return self.od[item]
        elif isinstance(item, int):
            return self.od[self.names[item]]
        return TypeError(f'invalid index type: {type(item)}')

    def __str__(self):
        return str(self.od)

    def __repr__(self):
        return str(self)
