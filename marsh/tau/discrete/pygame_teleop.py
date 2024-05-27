import pygame as pg, numpy as np

class PygameTeleop:
    keymap = {
        pg.K_a: [-1, 0],
        pg.K_s: [0, 1],
        pg.K_d: [1, 0],
        pg.K_w: [0, -1],
    }

    def __init__(self, target):
        anyvec = next(iter(self.keymap.values()))
        vecsize = len(anyvec)
        self.vec = np.zeros(vecsize)
        self._target = target
        self.keystate = dict.fromkeys(self.keymap.keys(), False)
    
    def __call__(self, event: 'pg.event.EventType | None'):
        try:
            if (event is None or event.type not in (pg.KEYUP, pg.KEYDOWN) 
                              or event.key not in self.keymap):
                return False
            vec = self.keymap[event.key]
            if event.type == pg.KEYDOWN and not self.keystate[event.key]:
                self.vec += vec
                self.keystate[event.key] = True
            elif event.type == pg.KEYUP and self.keystate[event.key]:
                self.vec -= vec
                self.keystate[event.key] = False
            return True
        finally:
            self._target(self.vec.copy())