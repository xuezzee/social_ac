import argparse
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal
from tensorboardX import SummaryWriter

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="Pendulum-v0")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=2, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = gym.make(args.env_name)

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './exp' + script_name + args.env_name +'./'

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=100):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device='cpu'):
        super(Actor, self).__init__()

        self.device = device
        self.conv = nn.Conv2d(
            in_channels=state_dim[2],
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=(1, 1)
        )
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=3,
        )
        self.input_size = int(5*5*self.conv.out_channels)
        self.l1 = nn.Linear(self.input_size, 32)
        self.l2 = nn.Linear(32, 32)
        self.LSTM = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
        )
        self.l_out = nn.Linear(32, action_dim)

        self.max_action = max_action

    def CNN_preprocess(self, input):
        x = torch.relu(self.conv(input))
        x = self.pool(x)
        x = x.flatten(start_dim=-3, end_dim=-1)
        return x

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        _, (x, _) = self.LSTM(x)
        x = torch.relu(x[0])
        x = torch.sigmoid(self.l_out(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, agent_num=5, device='cpu'):
        super(Critic, self).__init__()

        self.device = device
        # self.compressLayer = nn.Linear(state_dim, 32)
        self.l1 = nn.Linear(state_dim, 32)
        self.l2 = nn.Linear(32 , 32)
        self.l1_a = nn.Linear(action_dim, 32)
        self.l2_a = nn.Linear(32, 32)
        self.LSTM = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
        )
        self.l_out = nn.Linear(32, 1)


    # def forward(self, x, u):
    #     x = F.relu(self.l1(torch.cat([x, u], 1)))
    #     x = F.relu(self.l2(x))
    #     x = self.l3(x)
    #     return x

    def forward(self, s, a):
        x = torch.relu(self.l1(s))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.LSTM(x)[0][-1, :, :])
        a = torch.relu(self.l1_a(a))
        a = torch.relu(self.l2_a(a))
        x = torch.add(x, a)
        x = self.l_out(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, seq_len, max_action = None, device='cpu', num_agents=5):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.seq_len = seq_len
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(150*self.num_agents, action_dim*self.num_agents).to(device)
        self.critic_target = Critic(150*self.num_agents, action_dim*self.num_agents).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        # self.writer = SummaryWriter(directory)

        self.lr_scheduler = {
            'optC':torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=2000, gamma=0.99, last_epoch=-1),
            'optA':torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=2000, gamma=0.99, last_epoch=-1)
        }
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def get_global_optimizer_network(self, opt, net, lr_scheduler):
        self.critic_optimizer = opt[0]
        self.actor_optimizer = opt[1]
        self.lr_scheduler = {
            'optC':lr_scheduler['optC'],
            'optA':lr_scheduler['optA'],
        }
        self.gactor = net[0]
        self.gcritic = net[1]

    def CNN_preprocess(self, state):
        original_state = state.shape[0:2]
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state)
        state = state.flatten(0,1)
        state = self.actor.CNN_preprocess(state)
        return state.reshape(list(original_state)+[-1])

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy()

    def update(self):
        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(50)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # print("!!!!!!!")
            def transform_state(state):
                state = state.permute(1,0,2,3)
                state = state.reshape((self.seq_len, -1, state.shape[-1]))
                return state

            # Compute the target Q value
            action = action.flatten(start_dim=0, end_dim=1)
            state = transform_state(state)
            next_state = transform_state(next_state)

            actions = self.actor_target(next_state).reshape(-1, self.num_agents*self.action_dim)
            action = action.reshape(-1, self.num_agents*self.action_dim)
            next_state_copy = next_state.reshape(self.seq_len, -1, 150*self.num_agents).clone()
            state_copy = state.reshape(self.seq_len, -1, 150*self.num_agents).clone()
            # print()
            # next_state =
            target_Q = self.critic_target(next_state_copy, actions)
            target_Q = reward + (args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state_copy, action).to(self.device)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            for lp, gp in zip(self.critic.parameters(), self.gcritic.parameters()):
                gp._grad = lp.grad
            self.critic_optimizer.step()
            self.critic.load_state_dict(self.gcritic.state_dict())
            self.lr_scheduler['optC'].step()

            # Compute actor loss
            actor_loss = -self.critic(state_copy, self.actor(state).reshape(-1, self.num_agents*self.action_dim)).mean()
            # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            for lp, gp in zip(self.actor.parameters(), self.gactor.parameters()):
                gp._grad = lp.grad
            self.actor_optimizer.step()
            self.actor.load_state_dict(self.gactor.state_dict())
            self.lr_scheduler['optA'].step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def mask_action(self, mask, act):
        if not isinstance(mask, torch.Tensor):
            mask = torch.Tensor(mask)

        test = [torch.mul(mask[i], act[i]) for i in range(len(act))]
        test2 = [torch.mul(mask[i], act[i])[0].sum() for i in range(len(act))]
        masked_action = torch.cat([torch.mul(mask[i], act[i])/torch.mul(mask[i], act[i])[0].sum() for i in range(len(act))])
        m = Categorical(masked_action)
        a = m.sample()
        return a.data.cpu().numpy()

    def update_glaw(self):
        pass

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def main():
    agent = DDPG(state_dim, action_dim, max_action)
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >= args.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        if args.load: agent.load()
        total_step = 0
        for i in range(args.max_episode):
            total_reward = 0
            step =0
            state = env.reset()
            for t in count():
                action = agent.choose_action(state)
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)
                env.render()

                next_state, reward, done, info = env.step(action)
                if args.render and i >= args.render_interval : env.render()
                agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))

                state = next_state
                if done:
                    break
                step += 1
                total_reward += reward
            total_step += step+1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            agent.update()
           # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            if i % args.log_interval == 0:
                agent.save()
    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()