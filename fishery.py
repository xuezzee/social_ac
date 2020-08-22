import numpy as np
import random
import copy
import tkinter


class Fishery:
    def __init__(self):
        self.n_agents = 2
        self.park_size = 10
        self.grid1 = np.zeros((self.park_size, self.park_size))
        self.grid2 = np.zeros((self.park_size, self.park_size))
        self.root = None
        self.spawn = 0

    def get_obs(self):
        grid1 = copy.deepcopy(self.grid1)
        grid2 = copy.deepcopy(self.grid2)
        per_idx = np.where(grid1 == 1)
        grid1[per_idx] = 0
        per_idx = np.where(grid2 == 1)
        grid2[per_idx] = 0
        return [grid1.reshape(self.park_size * self.park_size), grid2.reshape(self.park_size * self.park_size)]

    def reset(self):
        self.steps = 0
        self.spawn1 = 0
        self.spawn2 = 0
        initman = (3, 1)
        self.lifetime = {}
        self.grid1[initman] = 1
        self.grid2[initman] = 1
        for i in range(4):
            self.grid1 = self.spawn_one_fish(1, 2)
            self.grid2 = self.spawn_one_fish(2, 2)
        return self.get_obs()  # [grid1,grid2]

    def spawn_one_fish(self, part, fish):
        ## fish:2
        ## person:1
        for f in self.lifetime.keys():
            self.lifetime[f] += 1
        if fish == 2 and part == 1:
            self.spawn1 += 1
        elif fish == 2 and part == 2:
            self.spawn2 += 1
        if part == 1:
            part_grid = self.grid1
        else:
            part_grid = self.grid2
        zero_index = np.where(part_grid == 0)
        # print(zero_index)
        # print(part_grid)
        big_index = np.where(part_grid == 3)
        if len(big_index[0]) > 0 and self.steps % 10 == 0:
            pos_2 = random.randint(0, len(big_index[0]) - 1)
            coord_2 = (big_index[0][pos_2], big_index[1][pos_2])
            part_grid[coord_2] = 0
        pos = random.randint(0, len(zero_index[0]) - 1)
        coord = (zero_index[0][pos], zero_index[1][pos])
        self.lifetime[coord] = 0
        part_grid[coord] = fish
        return part_grid

    def grow_fish(self):
        fish_index1 = np.where(self.grid1 == 2)
        fish_index2 = np.where(self.grid2 == 2)
        if len(fish_index1[0]) != 0:
            # for idx in fish_index1[0]:
            rand1 = random.randint(0, len(fish_index1[0]) - 1)
            pos = (fish_index1[0][rand1], fish_index1[1][rand1])
            # if self.lifetime
            self.grid1[pos] = 0
            self.spawn_one_fish(2, 3)
        if len(fish_index2[0]) != 0:
            rand2 = random.randint(0, len(fish_index2[0]) - 1)
            pos = (fish_index2[0][rand2], fish_index2[1][rand2])
            self.grid2[pos] = 0
            self.spawn_one_fish(1, 3)
        return [self.grid1, self.grid2]

    def move_on(self, part, act):
        if part == 1:
            per_idx = np.where(self.grid1 == 1)
            self.grid1[per_idx] = 0
            grid = self.grid1
        else:
            per_idx = np.where(self.grid2 == 1)
            self.grid2[per_idx] = 0
            grid = self.grid2
        if act == 0:
            per_idx = [per_idx[0] - 1, per_idx[1]]
        elif act == 1:
            per_idx = [per_idx[0] + 1, per_idx[1]]
        elif act == 2:
            per_idx = [per_idx[0], per_idx[1] - 1]
        else:
            per_idx = [per_idx[0], per_idx[1] + 1]
        for j in range(2):
            if per_idx[j][0] < 0:
                per_idx[j] = np.array([0])
            elif per_idx[j][0] >= self.park_size:
                per_idx[j] = np.array([self.park_size - 1])
        if grid[per_idx[0][0], per_idx[1][0]] == 2:
            reward = 1
        elif grid[per_idx[0][0], per_idx[1][0]] == 3:
            reward = 3
        else:
            reward = 0
        grid[per_idx[0][0], per_idx[1][0]] = 1
        return reward

    def step(self, actions):
        self.steps += 1
        if self.steps % 8 == 0 and self.steps > 0:
            self.grow_fish()
        r1 = self.move_on(1, actions[0])
        r2 = self.move_on(2, actions[1])

        if self.steps % 3 == 0:
            gs = 4 - (len(np.where(self.grid1 == 2)[0]) + len(np.where(self.grid1 == 3)[0]))
            if gs > 0 and self.spawn1 < 15:
                for g in range(gs):
                    self.grid1 = self.spawn_one_fish(1, 2)
            else:
                pass

            gs = 4 - (len(np.where(self.grid2 == 2)[0]) + len(np.where(self.grid2 == 3)[0]))
            if gs > 0 and self.spawn2 < 15:
                for g in range(gs):
                    self.grid2 = self.spawn_one_fish(2, 2)
            else:
                pass
        return self.get_obs(), [r1, r2], [False, False], {}

    def _close_view(self):
        if self.root:
            self.root.destory()
            self.root = None
            self.canvas = None
        # self.done = True

    def render(self):
        scale = 30
        width = self.park_size*2*scale
        height = self.park_size*scale
        if self.root is None:
            self.root = tkinter.Tk()
            self.root.title("fish")
            self.root.protocol("WM_DELETE_WINDOW", self._close_view)
            self.canvas = tkinter.Canvas(self.root, width=width, height=height)
            self.canvas.pack()

        self.canvas.delete(tkinter.ALL)
        self.canvas.create_rectangle(0, 0, width, height)
        maps = np.concatenate((self.grid1, self.grid2), axis=1)

        def fill_cell(x, y, color):
            self.canvas.create_rectangle(
                x * scale,
                y * scale,
                (x + 1) * scale,
                (y + 1) * scale,
                fill=color
            )

        # print(maps)
        # maps = np.ones((10,10))
        for x in range(self.park_size):
            for y in range(self.park_size * 2):
                if maps[x, y] == 0:  # y>=self.park_size and maps[x,y]==0:
                    if y >= self.park_size:
                        fill_cell(x, y, "White")
                    else:
                        fill_cell(y, x, "Black")
                elif maps[x, y] == 1:
                    # print((x,y))
                    fill_cell(y, x, "Orange")
                elif maps[x, y] == 2:
                    # print((x,y))
                    fill_cell(y, x, "Blue")
                elif maps[x, y] == 3:
                    fill_cell(y, x, "Red")

        self.root.update()


import time

if __name__ == "__main__":
    fish = Fishery()
    obs = fish.reset()
    for i in range(300):
        # print(i)
        actions = [random.randint(0, 3), random.randint(0, 3)]
        obs_n, r_n, _, _ = fish.step(actions)
        fish.render()
        time.sleep(0.4)
