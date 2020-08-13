import numpy as np
import random
class Fishery:
    def __init__(self):
        self.n_agents = 2
        self.park_size = 5
        self.gap = 1
        self.grid = np.full((self.park_size,self.park_size*2+self.gap),'0',dtype = object) ## show fish and man
        initman = (3,1)
        self.grid[initman] = 'p'
        self.grid[initman[0],initman[1]+self.park_size+self.gap] = 'p'
        self.grid_matrix = np.zeros((self.park_size,self.park_size*2+self.gap)) ## only fish will be shown
        
    def reset(self):
        ## fish birth with random in [1..4,1..4]
        for i in range(3):
            xf = random.randint(1,4)
            yf = random.randint(1,4)
    def init_part(self,part_grid):
        initman = (3,1)
        self.grid = np.full((self.park_size,self.park_size),'0',dtype = object)
        self.grid[initman] = 'p'
        for i in range(3):
            xf = random.randint(1,4)
            yf = random.randint(1,4)
        return 
    def spawn_fish(self,grid):
        ### total fish:6
        



