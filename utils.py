import torch
import numpy as np
from torch import nn
from gym.spaces import Discrete, Box
from envs.SocialDilemmaENV.social_dilemmas.envir.cleanup import CleanupEnv
from parallel_env_process import envs_dealer
# from PGagent import IAC, Centralised_AC, Law
# from network import Centralised_Critic

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
            n_state_ = np.array([state for state in n_state_.values()])
        n_reward = np.array([reward for reward in n_reward.values()])
        return n_state_/255., n_reward, done, info

    def reset(self):
        n_state = self.env.reset()
        if self.flatten:
            return np.array([state.reshape(-1) for state in n_state.values()])/255.
        else:
            return np.array([state for state in n_state.values()])/255.

    def seed(self,seed):
        self.env.seed(seed)

    def render(self, filePath = None):
        self.env.render(filePath)

    @property
    def observation_space(self):
        return Box(0., 1., shape=(675,), dtype=np.float32)

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def num_agents(self):
        return self.env.num_agents

def make_parallel_env(n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = env_wrapper(CleanupEnv(num_agents=4))
            # env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env()
    return envs_dealer([get_env_fn(i) for i in range(n_rollout_threads)])


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

# class Social_Agents():
#     def __init__(self,agents,agentParam):
#         self.Law = social_agent(agentParam)
#         self.agents = agents
#         self.n_agents = len(agents)
#
#     def select_masked_actions(self, state):
#         actions = []
#         for i, ag in zip(range(self.n_agents), self.agents):
#             masks, prob_mask = self.Law.select_action(state[i])
#             self.Law.prob_social.append(prob_mask)  # prob_social is the list of masks for each agent
#             pron_mask_copy = prob_mask  # deepcopy(prob_mask)
#             action, prob_indi = ag.select_masked_action(state[i], pron_mask_copy)
#             self.Law.pi_step.append(prob_indi)  # pi_step is the list of unmasked policy(prob ditribution) for each agent
#             actions.append(action)
#         return actions
#
#     def update(self, state, reward, state_, action):
#         for agent, s, r, s_,a in zip(self.agents, state, reward, state_, action):
#             agent.update(s,r,s_,a)
#
#     def update_law(self):
#         self.Law.update(self.n_agents)
#
#     def push_reward(self, reward):
#         for i, ag in zip(range(self.n_agents), self.agents):
#             ag.rewards.append(reward[i])
#         self.Law.rewards.append(sum(reward))


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma, i):
    bs = [s[i][0] for s in bs]
    ba = [a[i] for a in ba]
    br = [r[i] for r in br]
    if done:
        v_s_ = 0.               # terminal
    else:
        # v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]
        v_s_ = lnet.forward(s_)[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()
    # ca = v_wrap(np.vstack(bs))
    ca = torch.stack(bs,0)
    loss = lnet.loss_func(
        ca,
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


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