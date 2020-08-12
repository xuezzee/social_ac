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

from MAAC.algorithms.attention_sac import AttentionSAC
from MAAC.utils.buffer import ReplayBuffer
from parallel_env_process import envs_dealer
from envtest import envSocialDilemma,envLift

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=False, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

##### env: gathering
'''
n_agents = 2
state_dim = 400
action_dim = 8
env = GatheringEnv(2,"default_small2")
torch.manual_seed(args.seed)
envfolder = "gathering/"
model_name = "gathering_social_share"
file_name = "save_weight/" +envfolder+ model_name
ifload = False
save_eps = 20
ifsave_model = True
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name,}
n_episode = 201
n_steps = 1000
line = 10
'''

##### env: Lift
CentQ = True
if CentQ:
    useCenCritc = True
n_agents = 5
height = 4

env = envLift(n_agents,height)
torch.manual_seed(args.seed)
envfolder = "elevator/"
model_name = "lift_social_ag5h4"#"centSumR_ag5h4_"#"lift_iac"
file_name = "save_weight/" +envfolder+ model_name
ifload = False
save_eps = 20
ifsave_model = True
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name}
n_episode = 201
n_steps = 200
state_dim = 4*height+1
action_dim = 3
line = 10

#### env cleanup
'''
n_agents = 5

env = envSocialDilemma("cleanup",n_agents)
torch.manual_seed(args.seed)
envfolder = "cleanup/"
model_name = "clean_social_"#"clean_centSumR_indi_"#"lift_iac"
file_name = "save_weight/" +envfolder+ model_name
ifload = False
save_eps = 20
ifsave_model = True
agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":ifload,"filename": file_name}
n_episode = 401
n_steps = 1000#200
state_dim = 675
action_dim = 9
line = 20
'''
def add_para(id):
    agentParam["id"] = str(id)
    return agentParam

def main():
    writer = SummaryWriter('runs/'+envfolder+model_name)
    multiPGCen = CenAgents([Centralised_AC(action_dim,state_dim,add_para(i),useLaw=True,useCenCritc=False,num_agent=n_agents) for i in range(n_agents)],state_dim,agentParam)  # create PGagents as well as a social agent
    ## useCenCritc: use centra critic for normal agent
    multiPG = Agents([IAC(action_dim,state_dim,add_para(i),useLaw=True,useCenCritc=True,num_agent=n_agents) for i in range(n_agents)])  # create PGagents as well as a social agent
    
    for i_episode in range(n_episode):
        n_state, ep_reward = env.reset(), 0  # reset the env
        #print(" lr .... ",multiPG.agents[0].optimizerA.param_groups[0]['lr'])
        for t in range(n_steps):
            if  int(i_episode/line)%2==0:#((int(i_episode/line))%2==1):
                ## pis: output prob(detach()) only
                pis = multiPG.choose_indi_probs(n_state)
                actions = multiPGCen.choose_masked_actions(n_state,pis)
            else:
                mask_probs = multiPGCen.choose_indi_probs(n_state)
                actions = multiPG.choose_masked_actions(n_state,mask_probs)     #select masked actions for every agent
            n_state_, n_reward, _, _ = env.step(actions)  # interact with the env
            if args.render and i_episode%30==0 and i_episode>0:  # render or not
                env.render()
            ep_reward += sum(n_reward)  # record the total reward
            if int(i_episode/line)%2==0:
                multiPGCen.update_share(n_state, n_reward, n_state_, actions)
            else:
                ## update_cent： update for centra normal agent
                multiPG.update_cent(n_state, n_reward, n_state_, actions)
                ## multiPG.update(n_state, n_reward, n_state_, actions)
            n_state = n_state_

        running_reward = ep_reward

        writer.add_scalar("ep_reward", ep_reward, i_episode)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            # logger.scalar_summary("ep_reward", ep_reward, i_episode)
        if i_episode % save_eps == 0 and i_episode > 11 and ifsave_model:
            multiPG.save(file_name)
            multiPGCen.save(file_name)
            #pass



if __name__ == '__main__':
    main()
