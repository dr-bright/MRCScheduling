import math
from pprint import pprint
import time
from typing import Any
import pygame as pg
import pygame.gfxdraw as pg_gfxdraw

import sys
from weakref import ref, proxy
from math import sin, cos
import numpy as np
import bisect

from tau import ContigBlock
from tau.contig.basics import PID, Sum
from tau.contig.pygame_teleop import PygameTeleop


FPS = 20


class Robot(ContigBlock):  # inherit to get access to .dt()
    charge = 400.0
    magnetic_coeff = 800.0
    mass = 1.0
    static_friction = 0.2
    dynamic_friction = 0.5
    force_cap = 500
    def __init__(self, pos, group, pid, label=None):
        self.target = None
        self.pid = pid
        self.pos = np.array(pos, float)
        self.vel = np.array([0,0], float)
        self.acc = np.array([0,0], float)
        self.applied_force = np.array([0,0], float)
        self.group = group
        self.label = label
        self.last_potential_force = None
    
    def render(self, canvas):
        pg_gfxdraw.filled_circle(canvas,
                                 round(self.pos[0]), round(self.pos[1]),
                                 20, (60, 60, 60))
        if self.label is not None:
            size = self.label.get_width(), self.label.get_height()
            size = np.array(size)
            pos = self.pos - size / 2
            canvas.blit(self.label, pos.astype(int))
    
    def physics(self, t):
        # use potential to avoid other charges
        # use pid to follow the target
        # pid controls robot acceleration vector
        # max accel is clipped
        # there is friction
        dt = self.dt(t)
        potential_force = np.array([0, 0], float)
        for other in self.group:
            if other is self:
                continue
            dir = self.pos - other.pos
            dir: np.ndarray
            distance_squared = sum(dir * dir)
            dir /= np.linalg.norm(dir)
            mag = self.charge * other.charge / distance_squared
            potential_force += dir * mag
        if self.last_potential_force is not None and dt >= 0.0001:
            df = (potential_force - self.last_potential_force) / dt
            self.last_potential_force = potential_force
            electric_dir = potential_force / np.linalg.norm(potential_force)
            magnetic_dir = np.array([*reversed(electric_dir)])
            magnetic_dir[0] *= -1
            magnetic_mag = self.magnetic_coeff * dt
            magnetic_force = magnetic_dir * magnetic_mag
            # print(magnetic_force)
            if not np.isnan(magnetic_force).any():
                potential_force += magnetic_force
        else:
            self.last_potential_force = potential_force
        applied_force = potential_force + self.applied_force
        # static friction is opposite to applied force
        # static friction is equal or less then applied force
        # static friction is no more than a certain limit
        applied_force_mag = np.linalg.norm(applied_force)
        static_friction_dir =  - applied_force / applied_force_mag
        static_friction_mag = min(self.static_friction, applied_force_mag)
        static_friction = static_friction_dir * static_friction_mag
        if np.isnan(static_friction).any():
            static_friction = np.zeros(2)
        # dynamic friction is proportional to the velocity
        dynamic_friction = - self.dynamic_friction * self.vel
        total_force  = applied_force + static_friction + dynamic_friction
        total_mag = np.linalg.norm(total_force)
        if total_mag > self.force_cap:
            total_force = total_force / total_mag * self.force_cap
        self.pos += (self.vel * dt).round(3)
        self.vel += (self.acc * dt).round(3)
        self.acc = (total_force / self.mass).round(3)
    
    def control(self, t):
        diff = np.zeros(2)
        if self.target is not None:
            diff[:] = self.target - self.pos
        self.applied_force[:] = self.pid(diff, t)


class TaskDispatcher:
    def __init__(self):
        self.task_bins = None
        self.active_tasks = None
    
    def fetch_schedule(self,
                       schedule: 'list[tuple[Any, float, Any]]',
                       start_time: float):
        # schedule = [ (start, agent, action) ]
        self.task_bins = task_bins = []
        agents = []
        task_bins: 'dict[int, list[int]]'
        # sorting_key = lambda task_id: schedule[task_id][-1]
        for task_id, task in enumerate(schedule):
            start, agent, action = task
            try:
                agent_id = agents.index(agent)
                task_bin = task_bins[agent_id]
            except ValueError:
                task_bin = []
                agent_id = len(agents)
                agents.append(agent)
                task_bins.append(task_bin)
            task = [start + start_time, None, agent, action]
            bisect.insort(task_bin, task)
        # now need to compute end times for each task
        for task_bin in task_bins:
            for i in range(len(task_bin) - 1):
                task_bin[i][1] = task_bin[i+1][0]
            task_bin[-1][1] = math.inf

    def dispatch(self, t):
        active_tasks = [None] * len(self.task_bins)
        for agent_id, task_bin in enumerate(self.task_bins):
            active_task_id = bisect.bisect_left(task_bin, [t])
            if active_task_id == 0:
                continue
            active_task_id -= 1
            active_task = task_bin[active_task_id]
            if active_task[1] <= t:
                continue
            active_tasks[agent_id] = active_task
        self.active_tasks = active_tasks
        return active_tasks


class LocationManager:
    location_size = 100

    def __init__(self, font: 'pg.font.Font | None' = None):
        # point, color, text
        self.locations = []
        self.font = font or pg.font.SysFont(pg.font.get_default_font(),
                                            16)
    
    def append(self, point, color, text):
        self.locations.append([point, color, text])
    
    def __getitem__(self, loc_id):
        return self.locations[loc_id][0]

    def render(self, screen: pg.surface.Surface):
        loc_size = np.array(self.location_size)
        if not loc_size.shape or loc_size.shape[0] == 1:
            loc_size = loc_size.reshape(1)[0]
            loc_size = np.array([loc_size] * 2)
        for point, color, text in self.locations:
            x, y = (point - loc_size / 2).astype(int)
            w, h = loc_size.astype(int)
            rect = (x, y, w, h)
            pg.draw.rect(screen, color, rect, 1)
            l = self.font.render(text, False, (0,0,0))
            size = np.array((l.get_width(), l.get_height()))
            text_loc = point - size / 2
            screen.blit(l, text_loc.astype(int))


def render_active_tasks(active_tasks: 'list[tuple[float, float, int, np.ndarray]]',
                        locations: 'LocationManager',
                        t: float, 
                        screen: pg.surface.Surface, 
                        font: 'pg.font.Font | None' = None):
    if font is None:
        font = pg.font.SysFont('Comic Sans', 16)
    for task in active_tasks:
        if task is None:
            continue
        start, end, robot_id, loc_id = task
        if loc_id > 0:
            loc_id -= 1
        point = locations[loc_id]
        progress = (t - start) / (end - start)
        red = np.array((255, 0, 0))
        green = np.array((0, 255, 0))
        background = (1 - progress) * red + progress * green
        foreground = 255 - background
        text = str(robot_id)
        label = font.render(text, False, foreground)
        pg_gfxdraw.filled_circle(screen, *point.astype(int),
                                 15, background)
        label_size = np.array((label.get_width(), label.get_height()))
        label_pos = point - label_size / 2
        screen.blit(label, label_pos.astype(int))


class GridLayout:
    def __init__(self, x, y, cell_width, cell_height):
        self.org = np.array((x, y))
        self.cell = np.array((cell_width, cell_height), float)

    def topleft(self, row, col):
        return (self.org + (row, col) * self.cell).astype(int)

    def center(self, row, col):
        return (self.org + (row + 0.5, col + 0.5) * self.cell).astype(int)


def create_scene(cell_size):
    # returns initialized location manager
    locations = LocationManager()
    grid_layout = GridLayout(0, 0, cell_size, cell_size)
    rob_color = (200, 200, 0)
    loc_color = (40, 200, 0)
    
    locations.append(grid_layout.center(1,3), loc_color, "Loc 1")
    locations.append(grid_layout.center(2,3), loc_color, "Loc 2")
    locations.append(grid_layout.center(3,3), loc_color, "Loc 3")
    locations.append(grid_layout.center(4,3), loc_color, "Loc 4")
    locations.append(grid_layout.center(5,3), loc_color, "Loc 5")
    
    locations.append(grid_layout.center(4,5), rob_color, "Rob 5")
    locations.append(grid_layout.center(3,5), rob_color, "Rob 4")
    locations.append(grid_layout.center(2,5), rob_color, "Rob 3")
    locations.append(grid_layout.center(4,1), rob_color, "Rob 2")
    locations.append(grid_layout.center(2,1), rob_color, "Rob 1")
    
    return locations
    

def test():
    pg.font.init()
    locations = create_scene(100)
    screen = pg.display.set_mode((800, 600))
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                running = False
        screen.fill((255, 255, 255))
        locations.render(screen)
        pg.display.flip()
        time.sleep(1 / 20)

def main():
    pg.font.init() # you have to call this at the start, 
                       # if you want to use this module.
    global_font = pg.font.SysFont('Comic Sans MS', 30)
    screen = pg.display.set_mode((1244, 700))
    robots: 'list[Robot]' = []
    locations = create_scene(100)  # [loc1-5, home5-1]
    for robot_id in range(1, 6):
        pid = PID(2, 0.3, 2, 100)
        pos = locations[-robot_id]
        label = global_font.render(str(robot_id), False, (255, 255, 255))
        robot = Robot(pos, robots, pid, label)
        robot.target = pos
        robots.append(robot)
    schedule = [
        # [start_time, robot, place]
        [29, 1,  5],
        [29, 2,  2],
        [29, 3,  3], # loc 3
        [29, 4,  4],
        [29, 5,  1],
        [42, 3, -3],
        [42, 4, -4], # home 4
        [50, 3,  4],
        [60, 5, -5],
        [60, 4,  1],
        [66, 1, -1],
        [66, 2, -2],
        [66, 3, -3],
        [66, 4, -4],
        [66, 5, -5],
    ]
    dispatcher = TaskDispatcher()
    t = time.time()
    dispatcher.fetch_schedule(schedule, t - 29 + 2)
    running = True
    dt = 1 / FPS
    while running:
        t = time.time()
        for event in pg.event.get():
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                running = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                dispatcher.fetch_schedule(schedule, t - 29 + 1)
                for robot_id, robot in enumerate(robots, 1):
                    robot.target = locations[-robot_id]
            elif event.type == pg.QUIT:
                running = False
        active_tasks = dispatcher.dispatch(t)
        for task in active_tasks:
            if task is None:
                continue
            # start_time, stop_time, robot, place
            _, _, robot_id, loc_id = task
            if loc_id > 0:
                loc_id -= 1
            robots[robot_id - 1].target = locations[loc_id]
        for rob in robots:
            rob.control(t)
            rob.physics(t)
        screen.fill((255,255,255))
        current_time_text = global_font.render(str(t), False, 16)
        screen.blit(current_time_text, (0,0))
        locations.render(screen)
        render_active_tasks(active_tasks, locations, t, screen, global_font)
        for rob in robots:
            rob.render(screen)
        pg.display.flip()
        time.sleep(dt)
        
if __name__ == '__main__':
    main()

