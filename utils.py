import torch
import numpy as np
from torch import nn
from gym.spaces import Discrete, Box
from envs.SocialDilemmaENV.social_dilemmas.envir.cleanup import CleanupEnv
from parallel_env_process import envs_dealer
import copy
import ray
# from PGagent import IAC, Centralised_AC, Law
# from network import Centralised_Critic

def make_parallel_env(n_rollout_threads, make_env, flatten, seed = 3):
    def get_env_fn(rank):
        def init_env():
            env = env_wrapper(make_env[0](num_agents=make_env[1]), flatten=False)
            # env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env()
    return envs_dealer([get_env_fn(i) for i in range(n_rollout_threads)])

class env_wrapper():
    def __init__(self,env,flatten=True):
        self.env = env
        self.flatten = flatten

    def step(self,actions,need_argmax=True):
        def action_convert(action,need_argmax):
            # action = list(action.values())
            act = {}
            for i in range(len(action)):
                if need_argmax:
                    act["agent-%d"%i] = np.argmax(action[i],0)
                else:
                    act["agent-%d"%i] = action[i]
            return act
        n_state_, n_reward, done, info = self.env.step(action_convert(actions,need_argmax))
        if self.flatten:
            n_state_ = np.array([state.reshape(-1) for state in n_state_.values()])
        else:
            n_state_ = np.array([state.reshape((-1,self.channel,self.width,self.height)) for state in n_state_.values()])
        n_reward = np.array([reward for reward in n_reward.values()])
        done = np.array([d for d in done.values()])
        return n_state_/255., n_reward, done, info

    def reset(self):
        n_state = self.env.reset()
        if self.flatten:
            return np.array([state.reshape(-1) for state in n_state.values()])/255.
        else:
            return np.array([state[np.newaxis,:,:,:].transpose(0,3,1,2) for state in n_state.values()])/255.
            # return np.array([state.reshape((-1,self.channel,self.width,self.height)) for state in n_state.values()])/255.

    def seed(self,seed):
        self.env.seed(seed)

    def render(self, filePath = None):
        self.env.render(filePath)

    @property
    def observation_space(self):
        if self.flatten:
            return Box(0., 1., shape=(675,), dtype=np.float32)
        else:
            return Box(0., 1., shape=(15,15,3), dtype=np.float32)

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def num_agents(self):
        return self.env.num_agents

    @property
    def width(self):
        if not self.flatten:
            return self.observation_space.shape[0]
        else: return None

    @property
    def height(self):
        if not self.flatten:
            return self.observation_space.shape[1]
        else: return None

    @property
    def channel(self):
        if not  self.flatten:
            return self.observation_space.shape[2]
        else: return None

class Agents():
    def __init__(self,agents,exploration=0.5):
        self.num_agent = len(agents)
        self.agents = agents
        self.exploration = exploration
        self.epsilon = 0.95


    def choose_action(self,state,is_prob=False):
        actions = {}
        agentID = list(state.keys())
        i = 0
        if is_prob:
            for agent, s in zip(self.agents, state.values()):
                actions[agentID[i]] = agent.choose_action(s/255.,is_prob).detach()
                i += 1
            return actions
        else:
            for agent, s in zip(self.agents, state.values()):
                actions[agentID[i]] = int(agent.choose_action(s.reshape(-1)/255.).cpu().detach().numpy())
                i += 1
            return actions

    def update(self, state, reward, state_, action):
        for agent, s, r, s_,a in zip(self.agents, list(state), list(reward), list(state_), list(action)):
            agent.update(s.reshape(-1)/255.,r,s_.reshape(-1)/255.,a)

    def save(self, file_name):
        for i, ag in zip(range(self.num_agent), self.agents):
            torch.save(ag.policy, file_name + "pg" + str(i) + ".pth")

def v_wrap(np_array, dtype=np.float32, device='cpu'):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array).to(device)

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

class Updater():
    def __init__(self, seq_len, device='cpu', n_agents=5):
        self.device = device
        self.seq_state = None
        self.seq_act_inf = None
        self.n_agents = n_agents
        self.seq_len = seq_len
        self.obs_batch = []
        self.act_batch = []
        self.next_obs_batch = []
        self.rew_batch = []

    def get_first_state(self, f_s):
        self.seq_state = [torch.zeros_like(torch.Tensor(f_s)).to(self.device) for i in range(self.seq_len-1)]
        self.seq_state.append(torch.Tensor(f_s).to(self.device))

    def get_first_act_inf(self, f_a):
        self.seq_act_inf = [torch.zeros_like(torch.Tensor(f_a)).to(self.device) for i in range(self.seq_len-1)]
        self.seq_act_inf.append(torch.Tensor(f_a).to(self.device))

    @property
    def get_new_state(self):
        pass

    @get_new_state.setter
    def get_new_state(self, n_s):
        self._roll(torch.Tensor(n_s).to(self.device), 's')

    @property
    def get_new_act_inf(self):
        pass

    @get_new_act_inf.setter
    def get_new_act_inf(self, n_a):
        self._roll(n_a, 'a')

    def _roll(self, new, s_a):
        if s_a == 's':
            self.seq_state.pop(0)
            self.seq_state.append(new)
        else:
            self.seq_act_inf.pop(0)
            self.seq_act_inf.append(new)

    def seq_obs(self, index):
        t = np.concatenate(self.seq_state, axis=1)
        return t[index]

    def seq_act(self):
        t = np.concatenate(self.seq_act_inf)
        return t

    def counter_acts(self, c_a, require_tensor):
        if require_tensor:
            return torch.cat(self.seq_act_inf[:-1] + [c_a]).to(self.device)
        else:
            return self.seq_act_inf[:-1] + [c_a]

    def get_next_seq_obs(self, next_obs, require_tensor):
        if isinstance(next_obs, torch.Tensor):
            if not require_tensor:
                next_obs = next_obs.cpu().numpy()
        else:
            if require_tensor:
                next_obs = torch.Tensor(next_obs).to(self.device)
        if require_tensor:
            return torch.cat(self.seq_state[1:] + [next_obs], axis=1).to(self.device)
        else:
            return self.seq_state[1:] + [next_obs]

    def get_next_innfluencer_act(self, next_act, require_tensor):
        if isinstance(next_act, torch.Tensor):
            if not require_tensor:
                next_act = next_act.cpu().numpy()
        else:
            if require_tensor:
                next_act = torch.Tensor(next_act).to(self.device)
        if require_tensor:
            return torch.cat(self.seq_act_inf[1:] + [next_act], axis=0).to(self.device)
        else:
            return self.seq_act_inf[1:] + [next_act]

    def push_and_pull(self, opt, lnet, gnet, done, s_, bs, ba, br, gamma, i):
        bs = [s[i].unsqueeze(1) for s in bs]
        ba = [a[i] for a in ba]
        br = [r[i] for r in br]
        if done:
            v_s_ = 0.               # terminal
        else:
            # v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]
            _, v_s_ = lnet.forward(s_.unsqueeze(1))
            v_s_ = v_s_.data.cpu().numpy()[0, 0]

        buffer_v_target = []
        for r in br[::-1]:    # reverse buffer r
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()
        # ca = v_wrap(np.vstack(bs))
        ca = torch.cat(bs,axis=1).to(self.device)
        loss = lnet.loss_func(
            ca,
            v_wrap(np.array(ba), dtype=np.int64, device=self.device) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba),device=self.device),
            v_wrap(np.array(buffer_v_target)[:, None],device=self.device))

        # calculate local gradients and push local parameters to global
        opt.zero_grad()
        loss.backward()
        for lp, gp in zip(lnet.parameters(), gnet.parameters()):
            gp._grad = lp.grad
        opt.step()

        # pull global parameters
        lnet.load_state_dict(gnet.state_dict())

    # def push_and_pull(self, opt, lnet, gnet, s_, gamma, i):
    #     bs = [obs[i] for obs in self.obs_batch]
    #     ba = [act[i] for act in self.act_batch]
    #     br = [rew[i] for rew in self.rew_batch]

def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )

import tensorflow as tf
import numpy as np
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(
                tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

class Runner():
    def __init__(self, env, n_agent, agents, episode=100, step=1000, logger=None):
        self.env = env_wrapper(env, flatten=False)
        self.logger = logger
        self.n_agent = n_agent
        self.agents = agents
        self.episode = episode
        self.step = step
        self.state_dim = agents[0].state_dim
        self.action_dim = agents[0].action_dim
        self.alpha = 0.15

    def run(self):
        x_s = 0
        ep = 0
        while ep < self.episode:
            if self.alpha <= 0.4 and ep % 2==0:
                self.alpha = self.alpha * 0.15
            else:
                self.alpha = 0.4
            # self.env.env.setAppleRespawnRate(ep)
            # total_step = 1
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = [0. for i in range(self.n_agent)]
            for step in range(1, 1001):
                # print(step)
                # print(ep)
                # if self.name == 'w00' and ep%10 == 0:
                #     path = "/Users/xue/Desktop/temp/temp%d"%ep
                #     if not os.path.exists(path):
                #         os.mkdir(path)
                #     self.env.render(path)
                if step == 1:
                    state_update = copy.deepcopy(state)
                    a0, prob0 = self.agents[0].choose_action(state[0], True)
                    a0_exe = [a0]
                    a0 = self.one_hot(self.action_dim, a0)
                else:
                    a0, prob0 = self.agents[0].choose_action(state_[0], True)
                    a0_exe = [a0]
                    a0 = self.one_hot(self.action_dim, a0)
                    state = state_
                actions = [self.agents[i].choose_action(state[i], True, a0[None, None, :]) for i in range(1, self.n_agent)]
                # actions = [self.agents[i].choose_action(state[i], True) for i in range(1, self.n_agent)]
                prob = [elem[1] for elem in actions]
                a_exe = a0_exe + [elem[0] for elem in actions]
                # actions = [a0] + [self.one_hot(self.action_dim, elem[0]) for elem in actions]

                self.env.render()

                state_, reward, done, _ = self.env.step(a_exe, need_argmax=False)


                ep_r = [ep_r[i] + reward[i] for i in range(self.n_agent)]
                x, _ = self._influencer_reward(reward[0], self.agents[1:], prob0, a0_exe, state[1:], prob, step)
                reward = [float(i) for i in reward]
                x_s += _
                reward[0] += x

                state_update_ = copy.deepcopy(state_)

                for agent, s, a, r, s_ in zip(self.agents, state_update, a_exe, reward, state_update_):
                    agent.update(s, r, s_, a)
                # buffer_a.append(a_exe)
                # buffer_s.append(s)
                # buffer_r.append(r)

                # if step % 5 == 0:  # update global and assign to local net
                #     _s0 = self.lnet[0].CNN_preprocess(v_wrap(s_[None, :]))
                #     a0 = self.lnet[0].choose_action(_s0, False)
                #     a0 = self.one_hot(self.action_dim, a0)
                #     _s = [torch.cat((self.lnet[i].CNN_preprocess(v_wrap(s_[i][None, :])), a0[None, :]), -1) for i in
                #           range(1, self.agent_num)]
                #     _s = [_s0] + _s
                #     # sync
                #     done = [False for i in range(self.agent_num)]
                #     [push_and_pull(self.opt[i], self.lnet[i], self.gnet[i], done[i],
                #                    _s[i], buffer_s, buffer_a, buffer_r, self.GAMMA, i)
                #      for i in range(self.agent_num)]
                #     [self.scheduler_lr[i].step() for i in range(self.agent_num)]
                #     buffer_s, buffer_a, buffer_r = [], [], []
                state_update = state_update_
                # total_step += 1
            print('ep%d' % ep, sum(ep_r), x_s)

            if self.logger != None:
                self.logger.scalar_summary("reward", sum(ep_r), ep)
                self.logger.scalar_summary("influence reward", x_s, ep)
            # if ep % 2 == 0:
            #     for agent in self.agents:
            #         if agent.temperature > 0.001: agent.temperature = agent.temperature * 0.5
            #         else: agent.temperature = 0.001
            x_s = 0
            ep += 1

    def _influencer_reward(self, e, nets, prob0, a0, state, p_a, step=0):
        p_cf = []
        counter_actions = [self.one_hot(self.action_dim, i)[None, None, :] for i in range(self.action_dim)]
        counter_actions.pop(a0[0])
        counter_prob = []
        for i in range(len(prob0[0])):
            if i != a0[0]:
                counter_prob.append(prob0[0][i])
        for i in range(len(nets)):
            y = nets[i].counterfactual(counter_actions, counter_prob)[0]
            x = p_a[i][0]
            p_cf.append(self.kl_div(x,y))
        return (1-self.alpha) * e + (self.alpha) * self._sum(p_cf), self._sum(p_cf)

    def _sum(self, tar):
        sum = 0
        for t in tar:
            sum += t
        return sum

    def one_hot(self, dim, index, Tensor=True):
        if Tensor:
            one_hot = torch.zeros(dim)
            one_hot[index] = 1.
            return one_hot
        else:
            one_hot = np.zeros(dim)
            one_hot[index] = 1.
            return one_hot

    def kl_div(self, p, q):
        """Kullback-Leibler divergence D(P || Q) for discrete probability dists

        Assumes the probability dist is over the last dimension.
        Taken from: https://gist.github.com/swayson/86c296aa354a555536e6765bbe726ff7
        p, q : array-like, dtype=float
        """
        p = np.asarray(p.cpu(), dtype=np.float)
        q = np.asarray(q.cpu(), dtype=np.float)

        return np.sum(np.where(p != 0, p * np.log(p / q), 0), axis=-1)

def influencer_reward(e, influencee, influencer_prob, influencer_act, states, influencee_prob, act_dim):
    def _sum(probs):
        sum = 0
        for p in probs:
            sum += p
        return sum

    def kl_div(self, p, q):
        """Kullback-Leibler divergence D(P || Q) for discrete probability dists

        Assumes the probability dist is over the last dimension.
        Taken from: https://gist.github.com/swayson/86c296aa354a555536e6765bbe726ff7
        p, q : array-like, dtype=float
        """
        p = np.asarray(p.cpu(), dtype=np.float)
        q = np.asarray(q.cpu(), dtype=np.float)

        return np.sum(np.where(p != 0, p * np.log(p / q), 0), axis=-1)

    def one_hot(self, dim, index, Tensor=True):
        if Tensor:
            one_hot = torch.zeros(dim)
            one_hot[index] = 1.
            return one_hot
        else:
            one_hot = np.zeros(dim)
            one_hot[index] = 1.
            return one_hot

    p_cf = []                                               #TODO
    counter_actions = [one_hot(act_dim, i)[None, None, :] for i in range(act_dim)]
    counter_actions.pop(influencer_act[0])
    counter_prob = []
    for i in range(len(influencer_prob[0])):
        if i != influencer_act[0]:
            counter_prob.append(prob0[0][i])
    for i in range(len(nets)):
        y = nets[i].counterfactual(counter_actions, counter_prob)[0]
        x = p_a[i][0]
        p_cf.append(self.kl_div(x, y))
    return (1 - self.alpha) * e + (self.alpha) * self._sum(p_cf), self._sum(p_cf)


def categorical_sample(probs, use_cuda=False):
    int_acs = torch.multinomial(probs, 1)
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    acs = torch.autograd.Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    return int_acs, acs

def create_seq_obs(seq, obs, l):   #TODO
    seq = np.roll(seq, l-1, axis=2)
    seq = np.concatenate((seq[1:], obs),axis=0)
    # t = tuple(range(3,len(seq.shape)))
    return seq


def plot(data, logger):
    def plot_per_step(d, name):
        for i in range(len(d)):
            logger.scalar_summary(name+':rew_per_step', d[i][0] ,d[i][1])
    l = len(data)
    reward = [data[i][0] for i in range(l)]
    ep = [data[i][1] for i in range(l)]
    name = [data[i][2] for i in range(l)]
    reward_per_step = [data[i][-1] for i in range(l)]
    [logger.scalar_summary(name[i], reward[i], ep[i]) for i in range(l)]
    [plot_per_step(reward_per_step[i], name[i]) for i in range(l)]
    mean = np.mean(reward)
    std = np.std(reward)
    # logger.scalar_summary('')
    logger.scalar_summary('sum', mean/std, ep[0])
    logger.scalar_summary('mean', mean, ep[0])
    logger.scalar_summary('std', std, ep[0])