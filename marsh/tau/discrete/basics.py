from .. import clip
import numpy as np

class Sum:
    def __init__(self, max=None, min=None, value=None):
        self.value = value
        self.max = max
        self.min = min if min is not None or max is None else -max

    def __call__(self, value):
        if self.value is None:
            self.value = np.array(value)
        else:
            self.value += value
        return clip(self.value, self.min, self.max, self.value)


class Diff:
    def __init__(self, max=None, min=None):
        self.last = None
        self.max = max
        self.min = min if min is not None or max is None else -max
    
    def __call__(self, value):
        if self.last is None:
            self.last = value
        output = value - self.last
        self.last = value
        return clip(output, self.min, self.max, output)


class PID:
    def __init__(self, p, i, d, max=None, min=None):
        self.p = np.array(p)
        self.i = np.array(i)
        self.d = np.array(d)
        if max is not None and min is None:
            min = -max
        self.sum = Sum(max, min)
        self.diff = Diff(max, min)
        self.max = max
        self.min = min
    
    def __call__(self, value: np.ndarray) -> np.ndarray:
        output = self.p.dot(value)
        output += self.sum(self.i.dot(value))
        output += self.diff(self.d.dot(value))
        return clip(output, self.min, self.max, output)


# TODO: think more about it
class ClosedLoop:
    def __init__(self, forward, backward):
        self.last = None
        self.forward = forward
        self.backward = backward
    
    def __call__(self, value: np.ndarray) -> np.ndarray:
        if self.last is None:
            self.last = value - value
        value += self.last
        output = self.forward(value)
        self.last = self.backward(output)
        return output

