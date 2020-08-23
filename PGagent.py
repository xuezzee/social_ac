import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from network import Actor, Critic, CNN_preprocess, Centralised_Critic, ActorLaw, A3CNet, CriticRNN, ActorRNN, A3CLaw
from utils import v_wrap, set_init, record, Logger, categorical_sample, Updater
import copy
import itertools
from collections import deque
from shared_adam import SharedAdam
from ddpg import DDPG
import random
import torchsnooper
import os
import scipy.stats

writer = Logger('./logsnx')


def change_order(input, i):
    input = [s.clone() for s in input]
    x = input.pop(i)
    return [x] + input

class IAC():
    def __init__(self, action_dim, state_dim, agentParam, useLaw, useCenCritc, num_agent, CNN=False, width=None,
                 height=None, channel=None):
        self.CNN = CNN
        self.device = agentParam["device"]
        if CNN:
            self.CNN_preprocessA = CNN_preprocess(width, height, channel)
            self.CNN_preprocessC = CNN_preprocess(width, height, channel)
            state_dim = self.CNN_preprocessA.get_state_dim()
        if agentParam["ifload"]:
            self.actor = torch.load(agentParam["filename"] + "indi_actor_" + agentParam["id"] + ".pth",
                                    map_location=torch.device('cuda'))
            self.critic = torch.load(agentParam["filename"] + "indi_critic_" + agentParam["id"] + ".pth",
                                     map_location=torch.device('cuda'))
        else:
            if useLaw:
                self.actor = ActorLaw(action_dim, state_dim).to(self.device)
            else:
                self.actor = Actor(action_dim, state_dim).to(self.device)
            if useCenCritc:
                self.critic = Centralised_Critic(state_dim, num_agent).to(self.device)
            else:
                self.critic = Critic(state_dim).to(self.device)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.noise_epsilon = 0.99
        self.constant_decay = 0.1
        self.optimizerA = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.optimizerC = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.lr_scheduler = {
            "optA": torch.optim.lr_scheduler.StepLR(self.optimizerA, step_size=100, gamma=0.92, last_epoch=-1),
            "optC": torch.optim.lr_scheduler.StepLR(self.optimizerC, step_size=100, gamma=0.92, last_epoch=-1)}
        if CNN:
            # self.CNN_preprocessA = CNN_preprocess(width,height,channel)
            # self.CNN_preprocessC = CNN_preprocess
            self.optimizerA = torch.optim.Adam(
                itertools.chain(self.CNN_preprocessA.parameters(), self.actor.parameters()), lr=0.0001)
            self.optimizerC = torch.optim.Adam(
                itertools.chain(self.CNN_preprocessC.parameters(), self.critic.parameters()), lr=0.001)
            self.lr_scheduler = {
                "optA": torch.optim.lr_scheduler.StepLR(self.optimizerA, step_size=10000, gamma=0.9, last_epoch=-1),
                "optC": torch.optim.lr_scheduler.StepLR(self.optimizerC, step_size=10000, gamma=0.9, last_epoch=-1)}
        # self.act_prob
        # self.act_log_prob

    # @torchsnooper.snoop()
    def choose_action(self, s):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        if self.CNN:
            s = self.CNN_preprocessA(s.reshape((1, 3, 15, 15)))
        self.act_prob = self.actor(s) + torch.abs(torch.randn(self.action_dim) * 0.05 * self.constant_decay).to(
            self.device)
        self.constant_decay = self.constant_decay * self.noise_epsilon
        self.act_prob = self.act_prob / torch.sum(self.act_prob).detach()
        m = torch.distributions.Categorical(self.act_prob)
        # self.act_log_prob = m.log_prob(m.sample())
        temp = m.sample()
        return temp

    def choose_act_prob(self, s):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        self.act_prob = self.actor(s, [], False)
        return self.act_prob.detach()

    def choose_mask_action(self, s, pi):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        if self.CNN:
            s = self.CNN_preprocessA(s.reshape((1, 3, 15, 15)))
        self.act_prob = self.actor(s, pi, True) + torch.abs(
            torch.randn(self.action_dim) * 0.05 * self.constant_decay).to(self.device)
        self.constant_decay = self.constant_decay * self.noise_epsilon
        self.act_prob = self.act_prob / torch.sum(self.act_prob).detach()
        m = torch.distributions.Categorical(self.act_prob)
        # self.act_log_prob = m.log_prob(m.sample())
        temp = m.sample()
        return temp

    def cal_tderr(self, s, r, s_, A_or_C=None):
        s = torch.Tensor(s).unsqueeze(0).to(self.device)
        s_ = torch.Tensor(s_).unsqueeze(0).to(self.device)
        if self.CNN:
            if A_or_C == 'A':
                s = self.CNN_preprocessA(s.reshape(1, 3, 15, 15))
                s_ = self.CNN_preprocessA(s_.reshape(1, 3, 15, 15))
            else:
                s = self.CNN_preprocessC(s.reshape(1, 3, 15, 15))
                s_ = self.CNN_preprocessC(s_.reshape(1, 3, 15, 15))
        v_ = self.critic(s_).detach()
        v = self.critic(s)
        return r + 0.9 * v_ - v

    def td_err_sn(self, s_n, r, s_n_):
        s = torch.Tensor(s_n).reshape((1, -1)).unsqueeze(0).to(self.device)
        s_ = torch.Tensor(s_n_).reshape((1, -1)).unsqueeze(0).to(self.device)
        v = self.critic(s)
        v_ = self.critic(s_).detach()
        return r + 0.9 * v_ - v

    def LearnCenCritic(self, s_n, r, s_n_):
        td_err = self.td_err_sn(s_n, r, s_n_)
        loss = torch.mul(td_err, td_err)
        self.optimizerC.zero_grad()
        loss.backward()
        self.optimizerC.step()
        self.lr_scheduler["optC"].step()

    def learnCenActor(self, s_n, r, s_n_, a):
        td_err = self.td_err_sn(s_n, r, s_n_)
        m = torch.log(self.act_prob[0][a])
        temp = m * td_err.detach()
        loss = -torch.mean(temp)
        self.optimizerA.zero_grad()
        loss.backward()
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

    def learnCritic(self, s, r, s_):
        td_err = self.cal_tderr(s, r, s_)
        loss = torch.square(td_err)
        self.optimizerC.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizerC.step()
        self.lr_scheduler["optC"].step()

    # @torchsnooper.snoop()
    def learnActor(self, s, r, s_, a):
        td_err = self.cal_tderr(s, r, s_)
        m = torch.log(self.act_prob[0][a])
        td_err = td_err.detach()
        temp = m * td_err[0][0]
        loss = -torch.mean(temp)
        self.optimizerA.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward(retain_graph=True)
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

    def update_cent(self, s, r, s_, a, s_n, s_n_):
        self.LearnCenCritic(s_n, r, s_n_)
        self.learnCenActor(s_n, r, s_n_, a)

    def update(self, s, r, s_, a):
        self.learnCritic(s, r, s_)
        self.learnActor(s, r, s_, a)

class Centralised_AC(IAC):
    def __init__(self, action_dim, state_dim, agentParam, useLaw, useCenCritc, num_agent):
        super().__init__(action_dim, state_dim, agentParam, useLaw, useCenCritc, num_agent)
        self.critic = None
        if agentParam["ifload"]:
            self.actor = torch.load(agentParam["filename"] + "law_actor_" + str(0) + ".pth",
                                    map_location=torch.device('cuda'))

    # def cal_tderr(self,s,r,s_):
    #     s = torch.Tensor(s).unsqueeze(0)
    #     s_ = torch.Tensor(s_).unsqueeze(0)
    #     v = self.critic(s).detach()
    #     v_ = self.critic(s_).detach()
    #     return r + v_ - v

    def learnActor(self, a, td_err):
        m = torch.log(self.act_prob[0][a]).to(self.device)
        temp = m * (td_err.detach()).to(self.device)
        loss = -torch.mean(temp)
        self.optimizerA.zero_grad()
        loss.backward()
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

    def update(self, s, r, s_, a, td_err):
        self.learnActor(a, td_err)

class A3C(mp.Process):
    def __init__(self, env, global_net, optimizer, global_ep, global_ep_r, res_queue, name, state_dim, action_dim,
                 agent_num, scheduler_lr):
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
                s_, r, done, _ = self.env.step(a, need_argmax=False)
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
            print('ep%d' % ep, self.name, sum(ep_r))
            ep += 1
            if self.name == "w00":
                self.sender.send([sum(ep_r), ep])
        self.res_queue.put(None)

class SocialInfluence(mp.Process):
    '''
    如果需要有某些agent先做决策，则将其放到agent list的最前面，将需要先做决策的agent的数
    量写入first_decision_num, take_action_in_turn改为True

    同时如果某些agent在采取动作的时候，需要基于其他agent的输出或者观测，需要实现最下面的
    modify_obs方法

    agent的返回的动作需要是形如（a(int), prob(类型根据自己需要决定)）的元组
    CNN_preprocess返回的是flatten的张量，该返回值会被存入buffer以供更新使用，
    critic网络不再使用CNN

    updater中会将state转换为时序序列，序列长度在初始化updater时定义，
    updater中的pull_and_pull为更新使用
    '''
    def __init__(self, env, global_net, optimizer, global_ep, global_ep_r, res_queue, name,
                 state_dim, action_dim, agent_num, scheduler_lr, multiProcess=True, device="cpu",
                 take_action_in_turn = False, first_decision_num = 0, glaw=None):
        if multiProcess:
            super(SocialInfluence, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.sender = None
        self.name = 'w%02i' % name
        self.agent_num = agent_num
        self.multiProcess = multiProcess
        self.GAMMA = 0.99
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt, self.opt_law = global_net, optimizer[:-1], optimizer[-1]
        self.glaw = glaw
        self.scheduler_lr = scheduler_lr
        self.law = A3CLaw(
                          glaw.s_dim,
                          glaw.a_dim,
                          device=device,
                         ).to(device)
        self.lnet = [A3CNet(
                            global_net[i].s_dim,
                            global_net[i].a_dim,
                            device=device,
                            ).to(device) for i in range(first_decision_num)]
        self.lnet = self.lnet + [A3CNet(
                                        global_net[i].s_dim,
                                        global_net[i].a_dim,
                                        device=device,
                                        ).to(device) for i in range(first_decision_num, agent_num)]
        self.env = env
        self.device = device
        self.updater = None
        self.take_inTurn = take_action_in_turn
        self.first_descision_num = first_decision_num

    def run(self):
        summary = [[0,0] for i in range(1000)]
        reward_pipe = [0 for i in range(1000)]
        switch = [True, 0]
        tot_step = 0
        last_ep_rew = 0
        x_s = 0
        ep = 0
        self.updater = Updater(4, self.device)
        while self.g_ep.value < 10000:
            s = self.env.reset()
            buffer_s, buffer_r = [], []
            buffer_a_int, buffer_a_logist, buffer_a_onehot = [], [], []

            '''updater需要根据初始state去初始化内部的时序序列'''
            self.updater.get_first_state(s)
            ep_r = [0. for i in range(self.agent_num)]
            for step in range(1, 1001):
                # print(step)
                if self.name == 'w00' and ep%10 == 0:
                    path = "/Users/xue/Desktop/record/record3/r%d"%ep
                    if not os.path.exists(path):
                        os.mkdir(path)
                    self.env.render(path)
                '''updater的seq_obs方法返回的是第i个agent对应的最新的时序序列，（RGB）格式基础上加seq'''
                s_pre = [self.lnet[i].CNN_preprocess(v_wrap(self.updater.seq_obs(i), device=self.device)) for i in range(self.first_descision_num)]
                a_preD = []
                if self.take_inTurn:
                    for i in range(self.first_descision_num):
                        a_preD.append(self.lnet[i].choose_action(s_pre[i][:, None, :], True))

                '''
                # use it in social influence
                if step == 1:
                    self.updater.get_first_act_inf(a0.unsqueeze(0))
                else:
                    self.updater.get_new_act_inf = a0.unsqueeze(0)
                '''

                s = [self.lnet[i].CNN_preprocess(v_wrap(self.updater.seq_obs(i), device=self.device)) for i in range(self.first_descision_num, self.agent_num)]
                s = self.modify_obs(None, seq_obs=s,)

                a = [self.lnet[i].choose_action(s[i].unsqueeze(1), False) for i in range(self.first_descision_num, self.agent_num)]

                a = a_preD + a
                a_int = [a[i][0] for i in range(self.agent_num)]
                a_logist = [a[i][1] for i in range(self.agent_num)]
                a_onehot = [self.one_hot(dim=self.action_dim, index=a_int[i], Tensor=True) for i in range(self.agent_num)]

                if self.glaw != None:
                    law_s = [self.law.CNN_preprocess(v_wrap(self.updater.seq_obs(i), device=self.device)) for i in range(self.first_descision_num, self.agent_num)]
                    a_int = [self.law.choose_action(law_s[i].unsqueeze(1), a_logist[i]) for i in range(self.first_descision_num, self.agent_num)]
                    mask = [mask[0] for mask in a_int]
                    a_int = [a[1] for a in a_int]

                if step % 500 == 0:
                    print('a_int:',a_int)

                s_, r, done, _ = self.env.step(a_int,need_argmax=False)
                ep_r = [ep_r[i] + r[i] for i in range(self.agent_num)]

                m = reward_pipe.pop(0)
                summed_reward = sum(r)
                last_ep_rew = last_ep_rew+summed_reward-m
                summary.pop(0)
                summary.append([last_ep_rew, tot_step, ])
                reward_pipe.append(summed_reward)
                tot_step += 1

                '''
                # use in social influence 
                x,_ = self._influencer_reward(r[0], self.lnet[1:], a_prob[0], a_int[0], s[1:], a_prob[1:], step)
                r = [float(i) for i in r]
                x_s += _.cpu().numpy()
                r[0] += x.detach().cpu().numpy()
                '''

                '''prob和onehot未使用，有需要可以使用'''
                if switch[0] or self.glaw == None:
                    buffer_a_int.append(a_int)
                    buffer_a_logist.append(mask)
                    buffer_a_onehot.append(a_onehot)
                    buffer_s.append(s)
                    buffer_r.append(r)
                else:
                    buffer_a_int.append(a_int)
                    buffer_a_logist.append(a_logist)
                    buffer_a_onehot.append(a_onehot)
                    buffer_s.append([torch.cat(change_order(law_s, i), dim=-1) for i in range(self.agent_num)])
                    buffer_r.append([sum(r) for i in r])

                _s = self.updater.get_next_seq_obs(s_, require_tensor=True)
                if switch[0] or self.glaw == None:
                    if step % 5 == 0:  # update global and assign to local net
                        if self.take_inTurn:
                            _s_pre = [self.lnet[i].CNN_preprocess(_s[i]) for i in range(self.first_descision_num)]
                            _a_pre = [self.lnet.choose_action(_s_pre[i][:, None, :], False)]
                            _s = [self.lnet[i].CNN_preprocess(_s[i]) for i in range(self.first_descision_num, self.agent_num)]
                            _s = _s_pre + _s
                        else:
                            _s = [self.lnet[i].CNN_preprocess(_s[i]) for i in range(self.agent_num)]

                        _s = self.modify_obs(None, seq_obs=_s)

                        # sync
                        done = [False for i in range(self.agent_num)]
                        [self.scheduler_lr[i].step() for i in range(self.agent_num)]
                        [self.updater.push_and_pull(self.opt[i], self.lnet[i], self.gnet[i], done[i], _s[i], buffer_s, buffer_a_int, buffer_r, buffer_a_logist, self.GAMMA, i) for i in range(self.agent_num)]
                        buffer_s, buffer_r = [], []
                        buffer_a_int, buffer_a_logist, buffer_a_onehot = [], [], []
                else:
                    if step % 5 == 0:
                        # print("update")
                        # print(a_law)
                        _s = [self.law.CNN_preprocess(_s[i]) for i in range(self.agent_num)]
                        loss = [self.updater.law_update(self.opt_law, self.law, self.glaw, done[i], torch.cat(change_order(_s, i), dim=1), buffer_s, buffer_a_int, buffer_r, buffer_a_logist,self.GAMMA, i) for i in range(self.agent_num)]
                        loss = sum(loss)
                        self.opt_law.zero_grad()
                        loss.backward()
                        for lp, gp in zip(self.law.parameters(), self.glaw.parameters()):
                            gp._grad = lp.grad
                        self.opt_law.step()
                        self.law.load_state_dict(self.glaw.state_dict())
                        buffer_s, buffer_r = [], []
                        buffer_a_int, buffer_a_logist, buffer_a_onehot = [], [], []

                s = s_
                self.updater.get_new_state = s
            switch[1] += 1
            if switch[1] == 3:
                switch[0] = not switch[0]
                switch[1] = 0
            print('ep%d'%ep, self.name, sum(ep_r), x_s)

            '''
            # use in social influence 
            x_s = 0
            '''
            ep+=1
            # if self.multiProcess:
            self.sender.send([sum(ep_r), ep, self.name, summary])
        if self.multiProcess:
            self.res_queue.put(None)

    def _influencer_reward(self, e, nets, prob0, a0, s, p_a, step=0):
        a_cf = []
        for i in range(self.action_dim):
            if i != a0[0]:
                a_cf.append(i)
        p_cf = []
        s_cf = []
        for i in range(self.agent_num-1):
            s_cf_i = []
            for j in range(self.action_dim-1):
                a = s[i][:,:,:-self.action_dim]
                b = self.updater.counter_acts(self.one_hot(self.action_dim, a_cf[j]).unsqueeze(0), require_tensor=True).unsqueeze(1)
                s_cf_i.append(torch.cat([a, b], -1).to(self.device))
            s_cf.append(s_cf_i)

        for i in range(len(nets)):
            t = torch.cat(s_cf[i],axis=1)
            temp = nets[i].choose_action(t, True)[1]
            _a = [temp[j] * prob0[0][a_cf[j]] for j in range(self.action_dim-1)]
            _a = self._sum(_a)/torch.sum(self._sum(_a))
            x = p_a[i][0]
            y = _a.detach()
            p_cf.append(torch.nn.functional.kl_div(torch.log(x),y,reduction="sum"))
        return 1*e + 0*self._sum(p_cf), self._sum(p_cf)

    def _sum(self, tar):
        sum = 0
        for t in tar:
            sum += t
        return sum

    def one_hot(self, dim, index, Tensor=True):
        '''
        将动作转换为onehot向量，index为int类型的动作，dim为action_dim
        '''
        if Tensor:
            one_hot = torch.zeros(dim)
            one_hot[index] = 1.
            return one_hot.to(self.device)
        else:
            one_hot = np.zeros(dim)
            one_hot[index] = 1.
            return one_hot

    def modify_obs(self, agents, seq_obs, **kwargs):
        '''
        put in agents that need to take as inputs the action or probability distribution of others
        and their original observation. returns the synthesis observation. Could be modified.
        inputs:
            --a list of agents
            --a function that returns the sequential observation
        '''
        return seq_obs


class IAC_RNN(IAC):
    '''
    This is the RNN version of Actor Crtic, in order to address the kind of problems where the temporal features are included
    '''
    def __init__(self,action_dim, state_dim, agentParam, useLaw, useCenCritc, num_agent, CNN=True, device='cpu', width=None,
                 height=None, channel=None, name=None):
        super().__init__(action_dim,state_dim, agentParam, useLaw, useCenCritc, num_agent)
        self.name = name
        self.device = device
        self.maxsize_queue = 3
        self.CNN = CNN
        self.width = width
        self.height = height
        self.channel = channel
        self.temperature = 0.001
        self.queue_s = deque([torch.zeros(state_dim).to(device).reshape(1,channel,width,height) for i in range(self.maxsize_queue)])
        self.queue_a = deque([torch.zeros(action_dim).to(device).reshape(1,1,action_dim) for i in range(self.maxsize_queue)])
        self.queue_s_update = deque([torch.zeros(state_dim).to(device).reshape(1,channel,width,height) for i in range(self.maxsize_queue)])
        # self.queue_cf = deque([torch.zeros(state_dim).reshape(1,9,1,state_dim) for i in range(self.maxsize_queue)])
        self.actor = ActorRNN(state_dim,action_dim,CNN).to(device)
        self.critic = CriticRNN(state_dim,action_dim,CNN).to(device)
        self.optimizerA = torch.optim.Adam(self.actor.parameters(),lr=0.001)
        self.optimizerC = torch.optim.Adam(self.critic.parameters(),lr=0.001)
        self.lr_scheduler = {
            "optA": torch.optim.lr_scheduler.StepLR(self.optimizerA, step_size=20000, gamma=0.9, last_epoch=-1),
            "optC": torch.optim.lr_scheduler.StepLR(self.optimizerC, step_size=20000, gamma=0.9, last_epoch=-1)}

    def collect_states(self, state):
        self.queue_s.pop()
        self.queue_s.insert(0, state)

    def collect_act_prob(self, action):
        self.queue_a.pop()
        self.queue_a.insert(0, action)

    def collect_state_update(self, state):
        self.queue_s_update.pop()
        self.queue_s_update.insert(0, state)

    def choose_action(self, s, is_prob=False, a=None):
        s = torch.Tensor(s).to(self.device)
        self.collect_states(s)
        if not isinstance(a, type(None)):
            a = torch.Tensor(a).to(self.device)
            self.collect_act_prob(a)
            self.queue_a.reverse()


        self.queue_s.reverse()
        if isinstance(a, type(None)):
            self.act_prob = self.actor(torch.cat(list(self.queue_s)).to(self.device))
        else:
            self.act_prob = self.actor(torch.cat(list(self.queue_s)).to(self.device),
                                       torch.cat(list(self.queue_a)).to(self.device).reshape((1, -1, self.action_dim)))
            self.queue_a.reverse()
        self.queue_s.reverse()


        # self.constant_decay = self.constant_decay*self.noise_epsilon
        # self.act_prob = self.act_prob/torch.sum(self.act_prob).detach()
        # print(self.name, " choose_action:", self.act_prob)
        if is_prob:
            m = torch.distributions.Categorical(self.act_prob)
            m = m.sample()
            return m.cpu().detach().numpy()[0], self.act_prob.detach()
        else:
            m = torch.distributions.Categorical(self.act_prob)
            m = m.sample()
            return m.detach().cpu().numpy()[0]

    def counterfactual(self, counter_act, counter_prob):
        def change_counter_action(act_queue, counter_act):
            act_queue[-1] = counter_act.to(self.device)
            return torch.cat(list(act_queue)).to(self.device)
        self.queue_s.reverse()
        action_queue_temp = copy.deepcopy(self.queue_a)
        action_queue_temp.reverse()
        # action_queue_temp = np.vstack(list(action_queue_temp))
        act_prob = [self.actor(torch.cat(list(self.queue_s)).to(self.device),
                               change_counter_action(action_queue_temp, act)) for act in counter_act]
        act_prob = [act_prob[i] * counter_prob[i] for i in range(self.action_dim-1)]
        self.queue_s.reverse()
        act_prob = sum(act_prob)
        # act_prob = sum(act_prob)/len(counter_act)
        act_prob = act_prob / sum(act_prob[0])
        return act_prob.cpu().detach()

    def cal_tderr(self, s, r, s_):
        s_ = torch.Tensor(s_).to(self.device)
        temp_q = copy.deepcopy(self.queue_s_update)
        temp_q.pop()
        temp_q.insert(0,s_)
        temp_q.reverse()
        # queue_s_update = self.CNN_preprocess(torch.cat(list(self.queue_s_update)), A_or_C="Critic")
        # temp_q = self.CNN_preprocess(torch.cat(list(temp_q)), A_or_C="Critic")
        v_ = self.critic(torch.cat(list(temp_q)).to(self.device)).detach()
        self.queue_s_update.reverse()
        v = self.critic(torch.cat(list(self.queue_s_update)).to(self.device))
        self.queue_s_update.reverse()
        return r + 0.99*v_ - v

    def update(self, s, r, s_, a):
        s = torch.Tensor(s).to(self.device)
        self.collect_state_update(s)
        td_err = self.learnCritic(s, r, s_)
        self.learnActor(s, r, s_, a, td_err)

    def learnCritic(self, s, r, s_):
        td_err = self.cal_tderr(s, r, s_)
        loss = torch.square(td_err)
        self.optimizerC.zero_grad()
        loss.backward()
        self.optimizerC.step()
        self.lr_scheduler["optC"].step()
        return  td_err.detach()

    # @torchsnooper.snoop()
    def learnActor(self, s, r, s_, a, td_err):
        def entropy(prob):
            entropy = 0
            for p in prob[0]:
                entropy -= p*torch.log(p)
            return entropy.to(self.device)
        # print(self.name, " learnActor:", self.act_prob)
        # td_err = self.cal_tderr(s, r, s_)
        m = torch.log(self.act_prob[0][a])
        td_err = td_err
        temp = m * (td_err[0][0]) + self.temperature * entropy(self.act_prob)
        loss = -torch.mean(temp)
        self.optimizerA.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

class influence_A3C():
    def __init__(self, obs_dim, act_dim, lr, agents, obs_type="RGB", width=None, height=None, channel=None, lr_scheduler=False, influencer_num=1, device="cpu"):
        self.agents = agents
        self.agent_num = len(agents)
        self.influencer_num = influencer_num
        self.lr_scheduler = lr_scheduler
        self.action = act_dim
        self.obs_type = obs_type
        self.device = device
        if obs_type == "RGB":self.width = width; self.height = height; self.channel = channel
        self.obs_dim = obs_dim
        # for i in range(self.agent_num):
            # self.agents[i].optimizer = SharedAdam(self.agents[i].parameters(), lr=lr, betas=(0.92,0.99))            #optimizer和scheduler放在agent（network）了
            # self.agents[i].optimizer = torch.optim.Adam(self.agents[i].parameters(), lr=lr)
            # if lr_scheduler:self.agents[i].lr_scheduler = torch.optim.lr_scheduler.StepLR(self.agents[i].optimizer, #SharedAdam是莫烦A3C中实现的optimizer，好像是用来同时更新两个网络的，细节不太懂
            #                                                                               step_size=10000,
            #                                                                               gamma=0.9,
            #                                                                               last_epoch=-1)


    def choose_influencer_action(self, observations):                               #选择influencer的action， 直接放obs
        influencer_act_logists = []
        influencer_act_prob = []
        influencer_act_int = []
        influencer_act_onehot = []
        for agent, obs in zip(self.agents[:self.influencer_num], observations[:self.influencer_num]):
            prob, logist = agent.choose_action(obs)
            int_act, act = categorical_sample(prob.detach())                                 #MAAC里的函数直接那过来的， 放入probability distribution, 返回int型动作和 onehot 动作
            influencer_act_logists.append(logist)
            influencer_act_prob.append(prob)
            influencer_act_onehot.append(act.detach().cpu().numpy())
            influencer_act_int.append(int_act)
        return influencer_act_onehot, influencer_act_prob, influencer_act_int, influencer_act_logists


    def choose_action(self, observations, influencer_action):                       #使用obs和influencer的动作作为输入
        if self.device == torch.device("cuda"):use_cuda = True
        else:use_cuda = False
        influencee_logist = []
        influencee_onehot = []

        for agent, obs in zip(self.agents[self.influencer_num:],
                                       observations[self.influencer_num:]):
            prob, logist = agent.choose_action(obs, influencer_action[0])                        #具体见network中A3CAgent
            int_act, act = categorical_sample(prob.detach(), use_cuda=use_cuda)
            influencee_logist.append(logist.detach())
            influencee_onehot.append(act.detach())
        return influencee_onehot, influencee_logist


    def update(self, samples):                                                      #两个网络同时更新，不确定是否管用
        obs, acs, rews, next_obs, dones, acls, inf_ac, = samples
        for agent, o, ac, rew, o_, i_ac in zip(self.agents, obs, acs, rews, next_obs, inf_ac):
            loss = agent.loss((o,ac,rew,o_,i_ac))
            agent.optimizerA.zero_grad()
            agent.optimizerC.zero_grad()
            loss.backward()
            agent.optimizerA.step()
            agent.optimizerC.step()
            # if self.lr_scheduler:
            #     agent.lr_scheduler.step()








