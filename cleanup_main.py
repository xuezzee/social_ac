import argparse
import torch
import numpy as np
import torch.multiprocessing as mp
from envs.SocialDilemmaENV.social_dilemmas.envir.cleanup import CleanupEnv
from envs.SocialDilemmaENV.social_dilemmas.envir.harvest import HarvestEnv
from PGagent import A3C
from MAAC.algorithms.attention_sac import AttentionSAC
from MAAC.utils.buffer import ReplayBuffer
from utils import env_wrapper, make_parallel_env, Logger
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
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# env = GatheringEnv(2)  # gym.make('CartPole-v1')
# env.seed(args.seed)
# torch.manual_seed(args.seed)

agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device}
# agentParam =

model_name = "pg_social"
file_name = "/Users/xue/Desktop/Social_Law/saved_weight/" + model_name
save_eps = 10
ifsave_model = True
logger = Logger('./logs')


def A3C_main():
    n_agents = 5
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    n_workers = torch.multiprocessing.cpu_count()
    n_workers = 3
    sender, recevier = mp.Pipe()
    global_net = [A3CNet(675, 9) for i in range(n_agents)]
    optimizer = [torch.optim.Adam(global_net[i].parameters(), lr=0.0001) for i in range(n_agents)]
    scheduler_lr = [torch.optim.lr_scheduler.StepLR(optimizer[i],step_size=1000000, gamma=0.9, last_epoch=-1) for i in range(n_agents)]
    envs = [env_wrapper(CleanupEnv(num_agents=n_agents)) for i in range(n_workers)]
    workers = [A3C(envs[worker], global_net, optimizer, global_ep, global_ep_r, res_queue, worker, 675, 9, n_agents, scheduler_lr) for worker in range(n_workers)]
    workers[0].sender = sender
    for worker in workers:
        worker.start()
    res = []
    while True:
        msg = recevier.recv()
        logger.scalar_summary("reward", msg[0], msg[1])
        # r = res_queue.get()
        # if r is not None:
        #     res.append(r)
        # else:
        #     break
    [w.join() for w in workers]

if __name__ == '__main__':
    A3C_main()