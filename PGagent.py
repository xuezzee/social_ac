import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from network import Actor,Critic,CNN_preprocess,Centralised_Critic,ActorLaw
import copy
import itertools
import random
import torchsnooper

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
