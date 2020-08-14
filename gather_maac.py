import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gather_env import GatheringEnv
from PGagent import  IAC,Centralised_AC
from network import Centralised_Critic
from copy import deepcopy
#from logger import Logger
from torch.utils.tensorboard import SummaryWriter
# from envs.ElevatorENV import Lift
from multiAG import CenAgents,Agents
from envtest import envSocialDilemma,envLift
from gym.spaces import Discrete, Box

from MAAC.algorithms.attention_sac import AttentionSAC
from MAAC.utils.buffer import ReplayBuffer
from parallel_env_process import envs_dealer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=False, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='interval between training status logs0 (default: 10)')
args = parser.parse_args()



n_agents = 2
state_dim = 400
action_dim = 8
env = GatheringEnv(2,"default_small2")
torch.manual_seed(args.seed)

# agentParam =

model_name = "gathering_maac"#"gathering_centIAC" #"gathering_social_v1"#gathering_1"
file_name = "save_weight/" + model_name
ifload = False
save_eps = 30
ifsave_model = True
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name}
n_episode = 151#201
n_steps = 1000
line = 15

## Lift env
CentQ = True
if CentQ:
    useCenCritc = True
'''
n_agents = 6
height = 4
print(" pure total_r ... ",useCenCritc, "agents number ... ",n_agents,"height ... ",height)
env = envLift(n_agents,height)
torch.manual_seed(args.seed)

model_name = "lift_gumbel_ag6h4"#"lift_iac"
file_name = "save_weight/" + model_name
ifload = False
save_eps = 50
ifsave_model = True
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name}
n_episode = 201#201
n_steps = 200
state_dim = 4*height+1
action_dim = 3
line = 50
'''

class env_wrapper():
    def __init__(self,env):
        self.env = env
    def step(self,actions):
        ## actions: one-hot [0,0,...1]
        def action_convert(actions):
            # action = list(action.values())
            act = []
            for i in range(len(actions)):
                act.append(np.argmax(actions[i],0))
            return act
        n_state_, n_reward, done, info = self.env.step(action_convert(actions))
        return n_state_, n_reward, done, info
    '''
    def step(self,actions):
        def action_convert(action):
            # action = list(action.values())
            act = {}
            for i in range(len(action)):
                act["agent-%d"%i] = np.argmax(action[i],0)
            return act
        n_state_, n_reward, done, info = self.env.step(action_convert(actions))
        n_state_ = np.array([state.reshape(-1) for state in n_state_.values()])
        n_reward = np.array([reward for reward in n_reward.values()])
        return n_state_/255., n_reward, done, info
    '''
    def reset(self):
        n_state = self.env.reset()
        return n_state#np.array([state.reshape(-1) for state in n_state.values()])/255.

    def seed(self,seed):
        self.env.seed(seed)

    def render(self):
        self.env.render()

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(state_dim,))

    @property
    def action_space(self):
        return Discrete(action_dim)

    @property
    def num_agents(self):
        return self.env.n_agents

#env.seed(args.seed)
def make_parallel_env(n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = env_wrapper(GatheringEnv(2,"default_small2"))#env_wrapper(CleanupEnv(num_agents=4))
            # env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env()
    # if n_rollout_threads == 1:
#         return DummyVecEnv([get_env_fn(0)])
#     else:
#         return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
    return envs_dealer([get_env_fn(i) for i in range(n_rollout_threads)])



def add_para(id):
    agentParam["id"] = str(id)
    return agentParam

def MAAC_main():
    writer = SummaryWriter('runs_maac/'+model_name)  
    n_rollout_threads = 12
    env = make_parallel_env(n_rollout_threads,3)
    agent_init_params = [{'num_in_pol': state_dim, 'num_out_pol': action_dim} for i in range(n_agents)]
    multiPG = AttentionSAC(agent_init_params, [[state_dim,action_dim] for a in range(n_agents)])  # create PGagents as well as a social agent
    multiPG = AttentionSAC.init_from_env(env,
                                       tau=0.001,
                                       pi_lr=0.001,
                                       q_lr=0.001,
                                       gamma=0.99,
                                       pol_hidden_dim=128,
                                       critic_hidden_dim=128,
                                       attend_heads=4,
                                       reward_scale=100.)
    replay_buffer = ReplayBuffer(10000,2,[state_dim]*n_agents,[action_dim]*n_agents)
    for i_episode in range(n_episode):
        #print("i_episode:",i_episode)
        n_state, ep_reward1, ep_reward2 = env.reset(), 0, 0  # reset the env
        # n_state2,ep_reward2 = env2.reset(), 0
        ep_reward5 = 0
        for t in range(n_steps):
            # n_state = np.array(list(n_state.values())).reshape(400,-1)
            # n_state2 = np.array(list(n_state2.values())).reshape(400, -1)
            # i1 = n_state[:,1]
            # i2 = np.vstack(n_state[:,1])
            #print(t)
            torch_obs = [torch.autograd.Variable(torch.Tensor(np.vstack(n_state[:, i])),
                                  requires_grad=False)
                         for i in range(multiPG.nagents)]
            torch_agent_actions = multiPG.step(torch_obs,explore=True)  # agent.select_action(state)   #select masked actions for every agent
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]
            # actions = multiPG.select_masked_actions(n_state)
            n_state_, n_reward, done, _ = env.step(actions)  # interact with the env
            # if args.render:  # render or not
            #     env.render()
            # plt.close()
            # multiPG.push_reward(n_reward)  # each agent receive their own reward, the law receive the summed reward
            ep_reward1 += sum(n_reward[0])  # record the total reward
            ep_reward2 += sum(n_reward[1])
            ep_reward5 += sum(n_reward[6])
            #ep_reward += sum(n_reward)
            #if i_episode==0:
                #print("n_reward ....  .... ",n_reward)
            replay_buffer.push(n_state, agent_actions, n_reward, n_state_, done)
            n_rollout_threads = 12
            t += n_rollout_threads
            use_gpu = False
            if (len(replay_buffer) >= 1024 and
                    (t % 100) < 12):
                if use_gpu:
                    multiPG.prep_training(device='gpu')
                else:
                    multiPG.prep_training(device='cpu')
                for u_i in range(4):
                    sample = replay_buffer.sample(1024,
                                                  to_gpu=use_gpu)
                    multiPG.update_critic(sample, logger=None)
                    multiPG.update_policies(sample, logger=None)
                    multiPG.update_all_targets()
                multiPG.prep_rollouts(device='cpu')
            # multiPG.update_law()
            n_state = n_state_

        # running_reward = ep_reward
        # loss = multiPG.update_agents()  # update the policy for each PGagent
        # multiPG.update_law()  # update the policy of law
        if i_episode % args.log_interval == 0:
            print('Episode {}\tAverage reward 1: {:.2f}\tAverage reward 2: {:.2f}\tAverage reward 3: {:.2f}'.format(
                i_episode, ep_reward1, ep_reward2,ep_reward5))
        writer.add_scalar("ep_reward", ep_reward1, i_episode)

            # logger.scalar_summary("ep_reward", ep_reward, i_episode)

        # if i_episode % save_eps == 0 and i_episode > 1 and ifsave_model:
        #     multiPG.save(file_name)
        # #

if __name__ == '__main__':
    MAAC_main()
