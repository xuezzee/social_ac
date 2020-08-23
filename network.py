import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import set_init
import itertools
import numpy as np
from torch.distributions import Categorical


# from gumble import gumbel_softmax


class CNN_preprocess(nn.Module):
    def __init__(self, width, height, channel):
        super(CNN_preprocess, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=(1, 1))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=5)
        self.Conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=3)

    def forward(self, x):
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Conv3(x)
        x = F.relu(x)
        return torch.flatten(x)

    def get_state_dim(self):
        return 64


class Actor(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()
        self.Linear1 = nn.Linear(state_dim, 128)
        self.Dropout1 = nn.Dropout(p=0.3)
        self.Linear2 = nn.Linear(128, 128)
        self.Dropout2 = nn.Dropout(p=0.3)
        self.Linear3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.Dropout1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = self.Dropout2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        return F.softmax(x)


class ActorLaw(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(ActorLaw, self).__init__()
        self.action_dim = action_dim
        self.Linear1 = nn.Linear(state_dim, 128)
        self.Dropout1 = nn.Dropout(p=0.3)
        self.Linear2 = nn.Linear(128, 128)
        self.Dropout2 = nn.Dropout(p=0.3)
        self.Linear3 = nn.Linear(128, action_dim)

    def forward(self, x, rule, flag):
        x = self.Linear1(x)
        x = self.Dropout1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = self.Dropout2(x)
        x = F.relu(x)
        y = self.Linear3(x)
        if flag == False:
            return y
        else:
            x = torch.mul(y, rule)
            x = F.softmax(x)
            return x

    '''
    def forward(self,x,rule,flag,mode,temp=1):
        x = self.Linear1(x)
        x = self.Dropout1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = self.Dropout2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        y = F.softmax(x)  ## normal
        mp = -x
        z = gumbel_softmax(mp,temp,self.action_dim)
        if flag == False:
            if mode == "normal":
                return y
            else:
                return z
        else:
            if mode == "normal":
                out = torch.mul(y,rule)
                return out
            else:
                out = torch.mul(z,rule)
                return out
    '''


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.Linear1 = nn.Linear(state_dim, 128)
        self.Dropout1 = nn.Dropout(p=0.3)
        self.Linear2 = nn.Linear(128, 128)
        self.Dropout2 = nn.Dropout(p=0.3)
        self.Linear3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.Dropout1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = self.Dropout2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        return x


class Centralised_Critic(nn.Module):
    def __init__(self, state_dim, n_ag):
        super(Centralised_Critic, self).__init__()
        self.Linear1 = nn.Linear(state_dim * n_ag, 128)
        self.Linear2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x


class A3CNet(nn.Module):
    '''
    该网络为最新A3C框架的基本网络，整合了actor和critic网络
    CNN_preprocess只用于actor，critic的输入为经过actor CNN处理后的张量
    '''
    def __init__(self, s_dim, a_dim, CNN=False, device='cpu'):
        super(A3CNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=(1, 1))
        # self.pi1 = nn.Linear(1,32)
        # self.pi2 = nn.Linear(32,32)
        self.LSTM_policy = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
        )
        self.LSTM_critic = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
        )
        self.pi1 = nn.Linear(s_dim, 32)
        self.pi2 = nn.Linear(32, 32)
        self.pi_out = nn.Linear(32, a_dim)
        self.v1 = nn.Linear(s_dim, 32)
        self.v2 = nn.Linear(32, 32)
        self.v_out = nn.Linear(32, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical
        self.device = device
        self.state_seq = []

    def CNN_preprocess(self, x, width=15, height=15):
        # x = x.reshape(-1,3,width,height)
        x = torch.relu(self.conv1(x))
        return torch.flatten(x, start_dim=-3, end_dim=-1)

    def forward(self, x):
        pi1 = torch.relu(self.pi1(x))
        pi2 = torch.relu(self.pi2(pi1))
        _, (pi3, _) = self.LSTM_policy(pi2)
        logits = self.pi_out(torch.relu(pi3[0]))
        v1 = torch.relu(self.v1(x))
        v2 = torch.relu(self.v2(v1))
        _, (v3, _) = self.LSTM_critic(v2)
        values = self.v_out(torch.relu(v3[0]))
        return logits, values

    def choose_action(self, s, dist=False):
        self.eval()
        logist, _ = self.forward(s)
        prob = F.softmax(logist, dim=1).data
        m = self.distribution(prob)
        if not dist:
            return m.sample().cpu().numpy()[0], logist
        else:
            return m.sample().cpu().numpy()[0], prob

    def loss_func(self, s, a, v_t, law):
        self.train()
        logist, values = self.forward(s.to(self.device))
        td = v_t - values
        c_loss = td.pow(2)
        logist = torch.add(logist, law.detach())

        probs = F.softmax(logist, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * (td.detach() + 0.001 * self.entropy(probs).detach())
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

    def entropy(self, distribution):
        entropy = []
        for distribution_per_step in distribution:
            sum = 0
            for prob in distribution_per_step:
                sum -= prob * torch.log(prob)
            entropy.append(sum.unsqueeze(0))
        return torch.cat(entropy).unsqueeze(0).to(self.device)


class A3CLaw(A3CNet):
    def __init__(self, s_dim, a_dim, device='cpu'):
        s_dim = 150
        super().__init__(s_dim, a_dim, device='cpu')
        self.Pool = nn.MaxPool2d(
            kernel_size=3,
            stride=3,
        )
        self.v1 = nn.Linear(5 * s_dim, 32)

    def CNN_preprocess(self, x, width=15, height=15):
        # x = x.reshape(-1,3,width,height)
        x = torch.relu(self.conv1(x))
        x = self.Pool(x)
        return torch.flatten(x, start_dim=-3, end_dim=-1)

    def forward(self, x):
        pi1 = torch.relu(self.pi1(x))
        pi2 = torch.relu(self.pi2(pi1))
        _, (pi3, _) = self.LSTM_policy(pi2)
        logits = self.pi_out(torch.relu(pi3[0]))
        # v1 = torch.relu(self.v1(x))
        # v2 = torch.relu(self.v2(v1))
        # _, (v3, _) = self.LSTM_critic(v2)
        # values = self.v_out(v3[0])
        return logits

    def value(self, x):
        v1 = torch.relu(self.v1(x))
        v2 = torch.relu(self.v2(v1))
        _, (v3, _) = self.LSTM_critic(v2)
        values = self.v_out(torch.relu(v3[0]))
        return values

    def choose_action(self, s, agentsAct, dist=False):
        self.eval()
        logist = self.forward(s)
        logist = torch.add(torch.tanh(logist), agentsAct.detach())
        prob = F.softmax(logist, dim=1).data
        # prob = torch.mul(prob, agentsAct)
        # prob = prob/sum(prob)
        m = self.distribution(prob)
        if not dist:
            return logist.data.numpy(), m.sample().cpu().numpy()[0]
        else:
            return m.sample().cpu().numpy()[0], prob

    def loss_func(self, s, a, v_t, agentsAct):
        self.train()
        logits = self.forward(s[:,:,:self.s_dim].to(self.device))
        values = self.value(s)
        logits = torch.add(torch.tanh(logits), agentsAct)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * (td.detach() + 0.001 * self.entropy(probs).detach())
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class A3CAgent(nn.Module):
    def __init__(self, act_dim, width, height, channel=3, gamma=0.9, influencer=False, seq_len=5, device="cpu"):
        super(A3CAgent, self).__init__()
        self.convA = nn.Conv2d(in_channels=channel, out_channels=6, kernel_size=3, stride=1, padding=(1, 1))
        self.convC = nn.Conv2d(in_channels=channel, out_channels=6, kernel_size=3, stride=1, padding=(1, 1))
        if influencer:
            act_dim_input = 0
        else:
            act_dim_input = act_dim
        self.influencer = influencer
        self.input_size = self.convA.out_channels * width * height + act_dim_input
        self.critic_size = self.convC.out_channels * width * height
        self.policy = nn.Sequential(
            nn.Linear(self.input_size, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.LSTM(input_size=32,
                    hidden_size=32,
                    num_layers=1)
        )
        self.policy_out = nn.Linear(32, 32)
        self.critic = nn.Sequential(
            nn.Linear(self.critic_size, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.LSTM(input_size=32,
                    hidden_size=32,
                    num_layers=1)
        )
        self.critic_out = nn.Linear(32, 1)
        self.logist = None
        self.optimizerA = torch.optim.Adam(itertools.chain(self.convA.parameters(), self.policy.parameters(), self.policy_out.parameters()), lr=0.001)
        self.optimizerC = torch.optim.Adam(itertools.chain(self.convC.parameters(), self.policy.parameters(), self.critic_out.parameters()), lr=0.001)
        self.lr_scheduler = None
        self.seq_len = seq_len
        self.gamma = gamma
        self.channel = channel
        self.width = width
        self.height = height
        self.device = device

    def choose_action(self, input, act=None, train=False):  # LSTM暂时用不了，如果要跑的话去掉LSTM，换成应该Linear
        if train:
            self.train()
        else:
            self.eval()

        x = self.convA(input)
        x = x.flatten(start_dim=1, end_dim=-1)
        if not self.influencer:
            x = torch.cat((x, act), dim=1)
        _, (x,_) = self.policy(x.reshape((self.seq_len, -1, self.input_size)))  # change if the size of cell changed
        x = self.policy_out(torch.relu(x[0]))
        prob = torch.softmax(x, -1)
        logist = torch.log_softmax(x, -1)
        return prob, logist

    def value(self, input):  # TODO
        # self.train()
        x = self.convC(input.reshape(-1, self.channel, self.width, self.height))
        _, (x, _) = self.critic(torch.flatten(x, start_dim=1, end_dim=-1).reshape(self.seq_len, -1, self.critic_size))
        v = self.critic_out(torch.relu(x[0]))
        return v

    def loss(self, sample):
        # self.train()
        obs, acs, rews, next_obs, inf_ac = sample
        index = torch.argmax(acs, -1)[:, None]
        acp, acls = self.choose_action(obs.flatten(start_dim=0, end_dim=1).to(self.device),
                                       inf_ac.flatten(start_dim=0, end_dim=1).to(self.device))
        logs = acls.gather(index=index,
                           dim=-1)  # 因为输入的是dim=9的log prob_distribution，所以用gather选择其中的一个log_prob，没debug，可能纬度对不上
        v = self.value(obs)
        v_ = self.value(next_obs)
        q = rews[:, None] + self.gamma * v_
        td_err = torch.add(-v, q)
        sq = torch.square(td_err)
        lossC = torch.square(td_err).mean()
        x = -torch.mul(logs, td_err.detach())
        lossA = -torch.mul(logs, td_err.detach()).mean()
        loss = lossA + lossC
        return loss

    def entropy(self, distribution):
        sum = 0
        for prob in distribution:
            sum -= prob * torch.log(prob)
        return sum


class ActorRNN(nn.Module):
    def __init__(self, state_dim, action_dim, CNN=True):
        super(ActorRNN, self).__init__()
        self.CNN = CNN
        if CNN:
            self.Conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=(1, 1))
            self.Linear_a = nn.Linear(action_dim, 32)
            self.Linear1 = nn.Linear(state_dim * 2, 32)
            self.Linear2 = nn.Linear(32, 32)
            self.LSTM = nn.LSTM(
                input_size=32,
                hidden_size=32,
                num_layers=1,
            )
            self.out = nn.Linear(32, action_dim)
        else:
            self.rnn = nn.GRU(
                input_size=state_dim,
                hidden_size=128,
                num_layers=1,
            )
            self.out = nn.Linear(128, action_dim)

    def CNN_preprocess(self, x):
        x = self.Conv1(x)
        x = torch.relu(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1).unsqueeze(0)
        return x

    def forward(self, x, a=None):
        x = self.CNN_preprocess(x)
        x = torch.relu(self.Linear1(x))
        x = torch.relu(self.Linear2(x))
        x = x.reshape(-1, 1, 32)
        if not isinstance(a, type(None)):
            a = torch.relu(self.Linear_a(a))
            a = a.reshape(-1, 1, 32)
            x = x + a
        x, h_n = self.LSTM(x, None)
        x = F.relu(x[-1, :, :])
        x = self.out(x)
        return F.softmax(x)


class CriticRNN(nn.Module):
    def __init__(self, state_dim, action_dim, CNN=True):
        super(CriticRNN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.CNN = CNN
        if CNN:
            self.Conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=(1, 1))
            self.Linear_a = nn.Linear(action_dim, 32)
            self.Linear1 = nn.Linear(state_dim * 2, 32)
            self.Linear2 = nn.Linear(32, 32)
            self.LSTM = nn.LSTM(
                input_size=32,
                hidden_size=32,
                num_layers=1,
            )
            self.out = nn.Linear(32, 1)
        else:
            self.rnn = nn.GRU(
                input_size=state_dim,
                hidden_size=128,
                num_layers=1,
            )
            self.out = nn.Linear(128, 1)

    def CNN_preprocess(self, x):
        if not isinstance(x, type(torch.Tensor())):
            x = torch.Tensor(x)
        x = self.Conv1(x)
        x = torch.relu(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1).unsqueeze(0)
        return x

    def forward(self, x, a=None):
        x = self.CNN_preprocess(x)
        x = torch.relu(self.Linear1(x))
        x = torch.relu(self.Linear2(x))
        x = x.reshape(-1, 1, 32)
        if not isinstance(a, type(None)):
            a = torch.relu(self.Linear_a(a))
            x = x + a
        x, h_n = self.LSTM(x, None)
        x = F.relu(x[-1, :, :])
        x = self.out(x)
        return x


if __name__ == "__main__":
    model_name = "pg_social"
    file_name = "train_para/" + model_name
    agentParam = {"ifload": True, "filename": file_name, "id": "0"}
    net = torch.load(agentParam["filename"] + "pg" + agentParam["id"] + ".pth", map_location=torch.device('cuda'))
    optimizer = optim.Adam(net.parameters(), lr=0.01)
