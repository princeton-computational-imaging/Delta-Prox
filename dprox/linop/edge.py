import numpy as np


class Edge(object):
    """The edge between two lin ops.
    """
    
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.data = None
        self.mag = None  # Used to get norm bounds.

    @property
    def size(self):
        return np.prod(self.data.shape)

    def __repr__(self) -> str:
        return f'Edge(id={id(self)}, start={self.start}, end={self.end}, data={self.data is not None})'