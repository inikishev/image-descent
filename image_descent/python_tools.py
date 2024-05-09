from typing import Optional, Any
from collections.abc import Sequence, Callable, Iterable

def flatten(iterable: Iterable) -> list[Any]:
    if isinstance(iterable, Iterable): return [a for i in iterable for a in flatten(i)]
    else: return [iterable]

class Compose:
    def __init__(self, *args): self.transforms = flatten(args)
    def __call__(self, x, *args, **kwargs):
        for t in self.transforms: 
            if t is not None: x = t(x, *args, **kwargs)
        return x