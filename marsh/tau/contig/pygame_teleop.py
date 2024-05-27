from ..discrete.pygame_teleop import PygameTeleop as DPygameTeleop
import pygame as pg

class PygameTeleop(DPygameTeleop):
    def __init__(self, target):
        self.dt = None
        def discrete_wrapper(value):
            target(value, self.t)
        super().__init__(discrete_wrapper)
    def __call__(self, event: pg.event.EventType, t: float):
        self.t = t
        return super().__call__(event)
    def tick(self, t: float):
        self._target(self.vec.copy(), t)