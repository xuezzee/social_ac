from multiprocessing import Process, Pipe
import numpy as np

def worker(env, pipe):
    # obs_space = env.observation_space
    while True:
        label, msg = pipe.recv()
        if label == "reset":
            state_init = env.reset()
            pipe.send(state_init)
        elif label == "step":
            s = env.step(msg)
            pipe.send(s)
        elif label == "render":
            env.render()
        elif label == "done":
            return

class envs_dealer():
    def __init__(self,envs):
        self.envs = envs
        self.num_processes = len(self.envs)
        self.pipe_container = [Pipe() for i in range(self.num_processes)]
        self.processes = [Process(target=worker, args=(self.envs[i], self.pipe_container[i][1]))
                                                            for i in range(self.num_processes)]
        for process in self.processes:
            process.start()

    def reset(self):
        for pipe in self.pipe_container:
            pipe[0].send(("reset", 0))
        return np.array([self.pipe_container[i][0].recv() for i in range(self.num_processes)])

    def step(self, actions):
        for i in range(self.num_processes):
            self.pipe_container[i][0].send(("step", actions[i]))
        ret = [self.pipe_container[i][0].recv() for i in range(self.num_processes)]
        state = [];reward = [];done = [];info = []
        for i in ret:
            state.append(i[0])
            reward.append(i[1])
            done.append(i[2])
            info.append(i[3])
        return np.array(state), np.array(reward), np.array(done), info

    def done(self):
        for pipe in self.pipe_container:
            pipe[0].send(("done", 0))

    def render(self):
        for pipe in self.pipe_container:
            pipe[0].send(("render", 0))

    @property
    def action_space(self):
        return [self.envs[0].action_space for i in range(self.envs[0].num_agents)]

    @property
    def observation_space(self):
        return [self.envs[0].observation_space for i in range(self.envs[0].num_agents)]