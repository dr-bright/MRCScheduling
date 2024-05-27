import numpy as np

def clip(arr, min=None, max=None, out=None, **kwargs):
    if min is None and max is None:
        if out is None:
            return np.copy(arr)
        return arr
    else:
        return np.clip(arr, min, max, out, **kwargs)


class ContigBlock:
    _last_time: float   # yeah, changes
    def dt(self, t: float):
        if not hasattr(self, '_last_time'):
            self._last_time = t
            return 0.0
        dt = t - self._last_time
        self._last_time = t
        return dt