from .. import ContigBlock, clip
from ..discrete.basics import Sum as DSum, Diff as DDiff, PID as DPID
import numpy as np


class Sum(DSum, ContigBlock):
    def __call__(self, value: np.ndarray, t: float):
        dt = self.dt(t)
        super().__call__(value * dt)

class Diff(DDiff, ContigBlock):
    def __call__(self, value: np.ndarray, t: float):
        dt = self.dt(t)
        super().__call__(value / dt)

class PID(DPID, ContigBlock):
    def __call__(self, value: np.ndarray, t: float):
        dt = self.dt(t)
        output = self.p.dot(value)
        output += self.i.dot(self.sum(value * dt))
        if dt >= 0.0001:
            output += self.d.dot(self.diff(value / dt))
        return clip(output, self.min, self.max, output)