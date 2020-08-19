import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from network import Actor, Critic, CNN_preprocess, Centralised_Critic, ActorLaw, A3CNet, CriticRNN, ActorRNN
from utils import v_wrap, set_init, push_and_pull, record, Logger, categorical_sample
import copy
import itertools
from collections import deque
from shared_adam import SharedAdam
import random
import torchsnooper
import os
import scipy.stats

writer = Logger('./logsn')


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
    def __init__(self, env, global_net, optimizer, global_ep, global_ep_r, res_queue, name, state_dim, action_dim, agent_num, scheduler_lr):
        super(SocialInfluence, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.sender = None
        self.name = 'w%02i' % name
        self.agent_num = agent_num
        self.GAMMA = 0.9
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = global_net, optimizer
        self.scheduler_lr = scheduler_lr
        self.lnet = [A3CNet(state_dim, action_dim)]
        self.lnet = self.lnet + [A3CNet(state_dim+1, action_dim) for i in range(1, agent_num)]
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
                s0 = s[0]
                a0, prob0 = self.lnet[0].choose_action(v_wrap(s0[None, :]), True)
                a0 = [a0]
                s = [np.concatenate((s[i],np.array(a0)),-1) for i in range(1, self.agent_num)]
                s = [s0] + s
                a = [self.lnet[i].choose_action(v_wrap(s[i][None, :]), True) for i in range(1, self.agent_num)]
                prob = [elem[1] for elem in a]
                a = a0 + [elem[0] for elem in a]
                s_, r, done, _ = self.env.step(a,need_argmax=False)
                # print(a)
                # if done[0]: r = -1
                ep_r = [ep_r[i] + r[i] for i in range(self.agent_num)]
                x = self._influencer_reward(r[0], self.lnet[1:], prob0, a0, s[1:], prob)
                r = [float(i) for i in r]
                r[0] += x.numpy()
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if step % 5 == 0 and step != 0:  # update global and assign to local net
                    _s0 = s_[0]
                    a0 = self.lnet[0].choose_action(v_wrap(_s0[None, :]), False)
                    a0 = [a0]
                    _s = [np.concatenate((s_[i], np.array(a0)), -1) for i in range(1, self.agent_num)]
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
            print('ep%d'%ep, self.name, sum(ep_r))
            ep+=1
            if self.name == "w00":
                self.sender.send([sum(ep_r),ep])
        self.res_queue.put(None)

    def _influencer_reward(self, e, nets, prob0, a0, s, p_a):
        a_cf = []
        for i in range(self.action_dim):
            if i != a0[0]:
                a_cf.append(i)
        p_cf = []
        s_cf = np.array([[np.concatenate((s[i][ :-1],np.array([a_cf[j]])),-1) for j in range(self.action_dim-1)] for i in range(self.agent_num-1)])
        for i in range(len(nets)):
            # _a = [nets[i].choose_action(v_wrap(s_cf[i][None, :]), True)[1] for i in range(self.agent_num-1)]
            # temp = nets[i].choose_action(v_wrap(s_cf[i][None, :]), True)[1][0]
            _a = [torch.mul(nets[i].choose_action(v_wrap(s_cf[i][None, :]), True)[1][0], prob0)]
            # _a =
            _a = torch.sum(_a[0],-2)
            x = p_a[i][0]
            y = _a.detach()
            p_cf.append(torch.nn.functional.kl_div(torch.log(x),y,reduction="sum"))
            # l = scipy.stats.entropy(x.numpy(), y.numpy())
        return e + 50*self._sum(p_cf)/len(p_cf)

    def _sum(self, tar):
        sum = 0
        for t in tar:
            sum += t
        return sum

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
    def __init__(self, obs_dim, act_dim, lr, agents, obs_type="RGB", width=None, height=None, channel=None, lr_scheduler=False, influencer_num=1):
        self.agents = agents
        self.agent_num = len(agents)
        self.influencer_num = influencer_num
        self.lr_scheduler = lr_scheduler
        self.action = act_dim
        self.obs_type = obs_type
        if obs_type == "RGB":self.width = width; self.height = height; self.channel = channel
        self.obs_dim = obs_dim
        for i in range(self.agent_num):
            self.agents[i].optimizer = SharedAdam(self.agents[i].parameters(), lr=lr, betas=(0.92,0.99))            #optimizer和scheduler放在agent（network）了
            if lr_scheduler:self.agents[i].lr_scheduler = torch.optim.lr_scheduler.StepLR(self.agents[i].optimizer, #SharedAdam是莫烦A3C中实现的optimizer，好像是用来同时更新两个网络的，细节不太懂
                                                                                          step_size=10000,
                                                                                          gamma=0.9,
                                                                                          last_epoch=-1)


    def choose_influencer_action(self, observations):                               #选择influencer的action， 直接放obs
        influencer_act_logists = []
        influencer_act_prob = []
        influencer_act_int = []
        influencer_act_onehot = []
        for agent, obs in zip(self.agents[:self.influencer_num], observations[:self.influencer_num]):
            prob, logist = agent.choose_action(obs)
            int_act, act = categorical_sample(prob)                                 #MAAC里的函数直接那过来的， 放入probability distribution, 返回int型动作和 onehot 动作
            influencer_act_logists.append(logist)
            influencer_act_prob.append(prob)
            influencer_act_onehot.append(act)
            influencer_act_int.append(int_act)
        return influencer_act_onehot, influencer_act_prob, influencer_act_int, influencer_act_logists


    def choose_action(self, observations, influencer_action):                       #使用obs和influencer的动作作为输入
        influencee_logist = []
        influencee_onehot = []
        for agent, obs, inf_act in zip(self.agents[self.influencer_num:],
                                       observations[self.influencer_num:],
                                       influencer_action[self.influencer_num:]):
            prob, logist = agent.choose_action(obs, inf_act)                        #具体见network中A3CAgent
            int_act, act = categorical_sample(prob)
            influencee_logist.append(logist)
            influencee_onehot.append(act)
        return influencee_onehot, influencee_logist


    def update(self, samples):                                                      #两个网络同时更新，不确定是否管用
        for agent, sample in zip(self.agents, samples):
            loss = agent.loss(sample)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            if self.lr_scheduler:
                agent.lr_scheduler.step()








