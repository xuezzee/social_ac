import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from network import Actor,Critic,CNN_preprocess,Centralised_Critic,ActorLaw, A3CNet
from utils import v_wrap, set_init, push_and_pull, record, Logger
import copy
import itertools
import random
import torchsnooper
import os
import scipy.stats


class IAC():
    def __init__(self,action_dim,state_dim,agentParam,useLaw,useCenCritc,num_agent,CNN=False, width=None, height=None, channel=None):
        self.CNN = CNN
        self.device = agentParam["device"]
        if CNN:
            self.CNN_preprocessA = CNN_preprocess(width,height,channel)
            self.CNN_preprocessC = CNN_preprocess(width,height,channel)
            state_dim = self.CNN_preprocessA.get_state_dim()
        if agentParam["ifload"]:
            self.actor = torch.load(agentParam["filename"]+"indi_actor_"+agentParam["id"]+".pth",map_location = torch.device('cuda'))
            self.critic = torch.load(agentParam["filename"]+"indi_critic_"+agentParam["id"]+".pth",map_location = torch.device('cuda'))
        else:
            if useLaw:
                self.actor = ActorLaw(action_dim,state_dim).to(self.device)
            else:
                self.actor = Actor(action_dim,state_dim).to(self.device)
            if useCenCritc:
                self.critic = Centralised_Critic(state_dim,num_agent).to(self.device)
            else:
                self.critic = Critic(state_dim).to(self.device)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.noise_epsilon = 0.99
        self.constant_decay = 0.1
        self.optimizerA = torch.optim.Adam(self.actor.parameters(), lr = 0.001)
        self.optimizerC = torch.optim.Adam(self.critic.parameters(), lr = 0.001)
        self.lr_scheduler = {"optA":torch.optim.lr_scheduler.StepLR(self.optimizerA,step_size=100,gamma=0.92,last_epoch=-1),
                             "optC":torch.optim.lr_scheduler.StepLR(self.optimizerC,step_size=100,gamma=0.92,last_epoch=-1)}
        if CNN:
            # self.CNN_preprocessA = CNN_preprocess(width,height,channel)
            # self.CNN_preprocessC = CNN_preprocess
            self.optimizerA = torch.optim.Adam(itertools.chain(self.CNN_preprocessA.parameters(),self.actor.parameters()),lr=0.0001)
            self.optimizerC = torch.optim.Adam(itertools.chain(self.CNN_preprocessC.parameters(),self.critic.parameters()),lr=0.001)
            self.lr_scheduler = {"optA": torch.optim.lr_scheduler.StepLR(self.optimizerA, step_size=10000, gamma=0.9, last_epoch=-1),
                                 "optC": torch.optim.lr_scheduler.StepLR(self.optimizerC, step_size=10000, gamma=0.9, last_epoch=-1)}
        # self.act_prob
        # self.act_log_prob
    #@torchsnooper.snoop()
    def choose_action(self,s):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        if self.CNN:
            s = self.CNN_preprocessA(s.reshape((1,3,15,15)))
        self.act_prob = self.actor(s) + torch.abs(torch.randn(self.action_dim)*0.05*self.constant_decay).to(self.device)
        self.constant_decay = self.constant_decay*self.noise_epsilon
        self.act_prob = self.act_prob/torch.sum(self.act_prob).detach()
        m = torch.distributions.Categorical(self.act_prob)
        # self.act_log_prob = m.log_prob(m.sample())
        temp = m.sample()
        return temp

    def choose_act_prob(self,s):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        self.act_prob = self.actor(s,[],False)
        return self.act_prob.detach()


    def choose_mask_action(self,s,pi):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        if self.CNN:
            s = self.CNN_preprocessA(s.reshape((1,3,15,15)))
        self.act_prob = self.actor(s,pi,True) + torch.abs(torch.randn(self.action_dim)*0.05*self.constant_decay).to(self.device)
        self.constant_decay = self.constant_decay*self.noise_epsilon
        self.act_prob = self.act_prob/torch.sum(self.act_prob).detach()
        m = torch.distributions.Categorical(self.act_prob)
        # self.act_log_prob = m.log_prob(m.sample())
        temp = m.sample()
        return temp
    def cal_tderr(self,s,r,s_,A_or_C=None):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        s_ = torch.Tensor(s_).unsqueeze(0).to(self.device)
        if self.CNN:
            if A_or_C == 'A':
                s = self.CNN_preprocessA(s.reshape(1,3,15,15))
                s_ = self.CNN_preprocessA(s_.reshape(1,3,15,15))
            else:
                s = self.CNN_preprocessC(s.reshape(1,3,15,15))
                s_ = self.CNN_preprocessC(s_.reshape(1,3,15,15))
        v_ = self.critic(s_).detach()
        v = self.critic(s)
        return r + 0.9*v_ - v

    def td_err_sn(self, s_n, r, s_n_):
        s = torch.Tensor(s_n).reshape((1,-1)).unsqueeze(0).to(self.device)
        s_ = torch.Tensor(s_n_).reshape((1,-1)).unsqueeze(0).to(self.device)
        v = self.critic(s)
        v_ = self.critic(s_).detach()
        return r + 0.9*v_ - v

    def LearnCenCritic(self, s_n, r, s_n_):
        td_err = self.td_err_sn(s_n,r,s_n_)
        loss = torch.mul(td_err,td_err)
        self.optimizerC.zero_grad()
        loss.backward()
        self.optimizerC.step()
        self.lr_scheduler["optC"].step()
    
    def learnCenActor(self,s_n,r,s_n_,a):
        td_err = self.td_err_sn(s_n,r,s_n_)
        m = torch.log(self.act_prob[0][a])
        temp = m*td_err.detach()
        loss = -torch.mean(temp)
        self.optimizerA.zero_grad()
        loss.backward()
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

    def learnCritic(self,s,r,s_):
        td_err = self.cal_tderr(s,r,s_)
        loss = torch.mul(td_err,td_err)
        self.optimizerC.zero_grad()
        loss.backward()
        self.optimizerC.step()
        self.lr_scheduler["optC"].step()
    #@torchsnooper.snoop()
    def learnActor(self,s,r,s_,a):
        td_err = self.cal_tderr(s,r,s_)
        m = torch.log(self.act_prob[0][a])
        temp = m*td_err.detach()
        loss = -torch.mean(temp)
        self.optimizerA.zero_grad()
        loss.backward()
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

    def update_cent(self,s,r,s_,a,s_n,s_n_):
        self.LearnCenCritic(s_n,r,s_n_)
        self.learnCenActor(s_n,r,s_n_,a)

    def update(self,s,r,s_,a):
        self.learnCritic(s,r,s_)
        self.learnActor(s,r,s_,a)



class Centralised_AC(IAC):
    def __init__(self,action_dim,state_dim,agentParam,useLaw,useCenCritc,num_agent):
        super().__init__(action_dim,state_dim,agentParam,useLaw,useCenCritc,num_agent)
        self.critic = None
        if agentParam["ifload"]:
            self.actor = torch.load(agentParam["filename"]+"law_actor_"+str(0)+".pth",map_location = torch.device('cuda'))

    # def cal_tderr(self,s,r,s_):
    #     s = torch.Tensor(s).unsqueeze(0)
    #     s_ = torch.Tensor(s_).unsqueeze(0)
    #     v = self.critic(s).detach()
    #     v_ = self.critic(s_).detach()
    #     return r + v_ - v

    def learnActor(self,a,td_err):
        m = torch.log(self.act_prob[0][a]).to(self.device)
        temp = m*(td_err.detach()).to(self.device)
        loss = -torch.mean(temp)
        self.optimizerA.zero_grad()
        loss.backward()
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

    def update(self,s,r,s_,a,td_err):
        self.learnActor(a,td_err)


class A3C(mp.Process):
    def __init__(self, env, global_net, optimizer, global_ep, global_ep_r, res_queue, name, state_dim, action_dim, agent_num, scheduler_lr):
        super(A3C, self).__init__()
        self.sender = None
        self.name = 'w%02i' % name
        self.agent_num = agent_num
        self.GAMMA = 0.9
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = global_net, optimizer
        self.scheduler_lr = scheduler_lr
        self.lnet = [A3CNet(state_dim, action_dim) for i in range(agent_num)]
        self.env = env

    def run(self):
        ep = 0
        while self.g_ep.value < 100:
            # total_step = 1
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = [0. for i in range(self.agent_num)]
            for step in range(1000):
                # print(ep)
                # if self.name == 'w00' and self.g_ep.value%10 == 0:
                #     path = "/Users/xue/Desktop/temp/temp%d"%self.g_ep.value
                #     if not os.path.exists(path):
                #         os.mkdir(path)
                #     self.env.render(path)
                a = [self.lnet[i].choose_action(v_wrap(s[i][None, :])) for i in range(self.agent_num)]
                s_, r, done, _ = self.env.step(a,need_argmax=False)
                # print(a)
                # if done[0]: r = -1
                ep_r = [ep_r[i] + r[i] for i in range(self.agent_num)]
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if step % 5 == 0:  # update global and assign to local net
                    # sync
                    done = [False for i in range(self.agent_num)]
                    [push_and_pull(self.opt[i], self.lnet[i], self.gnet[i], done[i],
                                   s_[i], buffer_s, buffer_a, buffer_r, self.GAMMA, i)
                                                    for i in range(self.agent_num)]
                    [self.scheduler_lr[i].step() for i in range(self.agent_num)]
                    buffer_s, buffer_a, buffer_r = [], [], []
                # if ep == 999:  # done and print information
                #     record(self.g_ep, self.g_ep_r, sum(ep_r), self.res_queue, self.name)
                #     break
                s = s_
                # total_step += 1
            print('ep%d'%ep, self.name, sum(ep_r))
            ep+=1
            if self.name == "w00":
                self.sender.send([sum(ep_r),ep])
        self.res_queue.put(None)

class SocialInfluence(mp.Process):
    def __init__(self, env, global_net, optimizer, global_ep, global_ep_r, res_queue, name, state_dim, action_dim, agent_num, scheduler_lr):
        super(SocialInfluence, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.sender = None
        self.name = 'w%02i' % name
        self.agent_num = agent_num
        self.GAMMA = 0.99
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = global_net, optimizer
        self.scheduler_lr = scheduler_lr
        self.lnet = [A3CNet(state_dim, action_dim)]
        self.lnet = self.lnet + [A3CNet(state_dim+action_dim, action_dim) for i in range(1, agent_num)]
        self.env = env

    def run(self):
        x_s = 0
        ep = 0
        while self.g_ep.value < 100:
            # total_step = 1
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = [0. for i in range(self.agent_num)]
            for step in range(1, 1000):
                # print(ep)
                if self.name == 'w00' and ep%10 == 0:
                    path = "/Users/xue/Desktop/temp/temp%d"%ep
                    if not os.path.exists(path):
                        os.mkdir(path)
                    self.env.render(path)
                s0 = s[0]
                a0, prob0 = self.lnet[0].choose_action(v_wrap(s0[None, :]), True)
                a0_exe = [a0]
                # print(a0_exe)
                a0 = self.one_hot(self.action_dim, a0)
                s = [np.concatenate((s[i],a0),-1) for i in range(1, self.agent_num)]
                s = [s0] + s
                a = [self.lnet[i].choose_action(v_wrap(s[i][None, :]), True) for i in range(1, self.agent_num)]
                prob = [elem[1] for elem in a]
                a_exe = a0_exe + [elem[0] for elem in a]
                a = [a0] + [self.one_hot(self.action_dim, elem[0]) for elem in a]
                s_, r, done, _ = self.env.step(a_exe,need_argmax=False)
                # print(a)
                # if done[0]: r = -1
                ep_r = [ep_r[i] + r[i] for i in range(self.agent_num)]
                x,_ = self._influencer_reward(r[0], self.lnet[1:], prob0, a0_exe, s[1:], prob, step)
                r = [float(i) for i in r]
                x_s += _.numpy()
                r[0] += x.detach().numpy()
                buffer_a.append(a_exe)
                buffer_s.append(s)
                buffer_r.append(r)

                if step % 5 == 0:  # update global and assign to local net
                    _s0 = s_[0]
                    a0 = self.lnet[0].choose_action(v_wrap(_s0[None, :]), False)
                    a0 = self.one_hot(self.action_dim, a0)
                    _s = [np.concatenate((s_[i], a0), -1) for i in range(1, self.agent_num)]
                    _s = [_s0] + _s
                    # sync
                    done = [False for i in range(self.agent_num)]
                    [push_and_pull(self.opt[i], self.lnet[i], self.gnet[i], done[i],
                                   _s[i], buffer_s, buffer_a, buffer_r, self.GAMMA, i)
                                                    for i in range(self.agent_num)]
                    [self.scheduler_lr[i].step() for i in range(self.agent_num)]
                    buffer_s, buffer_a, buffer_r = [], [], []
                # if ep == 999:  # done and print information
                #     record(self.g_ep, self.g_ep_r, sum(ep_r), self.res_queue, self.name)
                #     break
                s = s_
                # total_step += 1
            print('ep%d'%ep, self.name, sum(ep_r), x_s)
            x_s = 0
            ep+=1
            if self.name == "w00":
                self.sender.send([sum(ep_r),ep])
        self.res_queue.put(None)

    def _influencer_reward(self, e, nets, prob0, a0, s, p_a, step=0):
        a_cf = []
        for i in range(self.action_dim):
            # if i != a0[0]:
            a_cf.append(i)
        p_cf = []
        s_cf = np.array([[np.concatenate((s[i][ :-self.action_dim],self.one_hot(self.action_dim, a_cf[j])),-1)
                          for j in range(self.action_dim)] for i in range(self.agent_num-1)])
        for i in range(len(nets)):
            temp = nets[i].choose_action(v_wrap(s_cf[i][:, :]), True)[1]
            _a = [temp[j] * prob0[0][a_cf[j]] for j in range(self.action_dim)]
            # _a = [torch.mul(nets[i].choose_action(v_wrap(s_cf[i][None, :]), True)[1], prob0)]
            _a = self._sum(_a)
            x = p_a[i][0]
            y = _a.detach()
            p_cf.append(torch.nn.functional.kl_div(torch.log(x),y,reduction="sum"))
            # if step == 999:
            #     print("p_a:",p_a[i][0], "cf_a:",y)
            # print(self._sum(p_cf))
            # l = scipy.stats.entropy(x.numpy(), y.numpy())
        return 1.*e + 0.*self._sum(p_cf), self._sum(p_cf)

    def _sum(self, tar):
        sum = 0
        for t in tar:
            sum += t
        return sum

    def one_hot(self, dim, index):
        oh = np.zeros(dim)
        oh[index] = 1.
        return oh