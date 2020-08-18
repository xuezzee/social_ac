import argparse
import torch
import numpy as np
import torch.multiprocessing as mp
from envs.SocialDilemmaENV.social_dilemmas.envir.cleanup import CleanupEnv
from envs.SocialDilemmaENV.social_dilemmas.envir.harvest import HarvestEnv
from PGagent import A3C, SocialInfluence, IAC_RNN
from MAAC.algorithms.attention_sac import AttentionSAC
from MAAC.utils.buffer import ReplayBuffer
from utils import env_wrapper, make_parallel_env, Logger, Runner
# from logger import Logger
from network import A3CNet
from torch.utils.tensorboard import SummaryWriter

# from envs.ElevatorENV import Lift

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=1, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=True, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs0 (default: 10)')
args = parser.parse_args()

# env = GatheringEnv(2)  # gym.make('CartPole-v1')
# env.seed(args.seed)
# torch.manual_seed(args.seed)

agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":False,"filename": None}
# agentParam =

model_name = "pg_social"
file_name = "/Users/xue/Desktop/Social_Law/saved_weight/" + model_name
save_eps = 10
ifsave_model = True
logger = Logger('./logs5')


def A3C_main_multiProcess():
    n_agents = 5
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    n_workers = mp.cpu_count()
    n_workers = 3
    sender, recevier = mp.Pipe()
    global_net = [A3CNet(675*2, 8)]
    global_net = global_net + [A3CNet(675*2+8, 8) for i in range(n_agents-1)]
    optimizer = [torch.optim.Adam(global_net[i].parameters(), lr=0.001) for i in range(n_agents)]
    scheduler_lr = [torch.optim.lr_scheduler.StepLR(optimizer[i],step_size=10000, gamma=0.8, last_epoch=-1) for i in range(n_agents)]
    envs = [env_wrapper(HarvestEnv(num_agents=n_agents),flatten=True) for i in range(n_workers)]
    # workers = [A3C(envs[worker], global_net, optimizer, global_ep, global_ep_r, res_queue, worker, 675, 9, n_agents, scheduler_lr) for worker in range(n_workers)]
    workers = [SocialInfluence(envs[worker], global_net, optimizer, global_ep, global_ep_r, res_queue, worker, 675, 8, n_agents,
                   scheduler_lr) for worker in range(n_workers)]
    workers[0].sender = sender
    for worker in workers:
        worker.start()
    res = []
    while True:
        msg = recevier.recv()
        # logger.scalar_summary("reward", msg[0], msg[1])
        # r = res_queue.get()
        # if r is not None:
        #     res.append(r)
        # else:
        #     break
    [w.join() for w in workers]
def A3C_main():
    n_agents = 5
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    n_workers = mp.cpu_count()
    n_workers = 3
    sender, recevier = mp.Pipe()
    global_net = [A3CNet(675, 9)]
    global_net = global_net + [A3CNet(676, 9) for i in range(n_agents-1)]
    optimizer = [torch.optim.Adam(global_net[i].parameters(), lr=0.0001) for i in range(n_agents)]
    scheduler_lr = [torch.optim.lr_scheduler.StepLR(optimizer[i],step_size=1000000, gamma=0.9, last_epoch=-1) for i in range(n_agents)]
    envs = [env_wrapper(CleanupEnv(num_agents=n_agents),flatten=True) for i in range(n_workers)]
    # workers = [A3C(envs[worker], global_net, optimizer, global_ep, global_ep_r, res_queue, worker, 675, 9, n_agents, scheduler_lr) for worker in range(n_workers)]
    workers = [SocialInfluence(envs[worker], global_net, optimizer, global_ep, global_ep_r, res_queue, worker, 675, 9, n_agents,
                   scheduler_lr) for worker in range(n_workers)]
    workers[0].sender = sender
    for worker in workers:
        worker.start()
    res = []
    while True:
        msg = recevier.recv()
        # logger.scalar_summary("reward", msg[0], msg[1])
        # r = res_queue.get()
        # if r is not None:
        #     res.append(r)
        # else:
        #     break
    [w.join() for w in workers]

def A2C_main():
    n_agent = 5
    env = CleanupEnv(num_agents=n_agent)
    # action_dim = env.action_space
    # state_dim = env.observation_space
    agents = [IAC_RNN(9, 675, agentParam, useLaw=False, useCenCritc=False, num_agent=n_agent, device=device,
                      width=15, height=15, channel=3, name="agent%d"%i) for i in range(n_agent)]
    runner = Runner(env, n_agent, agents, logger=None)
    runner.run()

if __name__ == '__main__':
    # A3C_main()
    A2C_main()