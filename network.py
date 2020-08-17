
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import set_init
from torch.distributions import Categorical
#from gumble import gumbel_softmax


class CNN_preprocess(nn.Module):
    def __init__(self,width,height,channel):
        super(CNN_preprocess,self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=(1,1))
        self.Conv2 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=5,stride=5)
        self.Conv3 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=3)


    def forward(self,x):
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
    def __init__(self,action_dim,state_dim):
        super(Actor,self).__init__()
        self.Linear1 = nn.Linear(state_dim,128)
        self.Dropout1 = nn.Dropout(p=0.3)
        self.Linear2 = nn.Linear(128,128)
        self.Dropout2 = nn.Dropout(p=0.3)
        self.Linear3 = nn.Linear(128,action_dim)

    def forward(self,x):
        x = self.Linear1(x)
        x = self.Dropout1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = self.Dropout2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        return F.softmax(x)

class ActorLaw(nn.Module):
    def __init__(self,action_dim,state_dim):
        super(ActorLaw,self).__init__()
        self.action_dim = action_dim
        self.Linear1 = nn.Linear(state_dim,128)
        self.Dropout1 = nn.Dropout(p=0.3)
        self.Linear2 = nn.Linear(128,128)
        self.Dropout2 = nn.Dropout(p=0.3)
        self.Linear3 = nn.Linear(128,action_dim)

    def forward(self,x,rule,flag):
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
            x = torch.mul(y,rule)
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
    def __init__(self,state_dim):
        super(Critic,self).__init__()
        self.Linear1 = nn.Linear(state_dim, 128)
        self.Dropout1 = nn.Dropout(p=0.3)
        self.Linear2 = nn.Linear(128, 128)
        self.Dropout2 = nn.Dropout(p=0.3)
        self.Linear3 = nn.Linear(128,1)

    def forward(self,x):
        x = self.Linear1(x)
        x = self.Dropout1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = self.Dropout2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        return x

class Centralised_Critic(nn.Module):
    def __init__(self,state_dim,n_ag):
        super(Centralised_Critic,self).__init__()
        self.Linear1 = nn.Linear(state_dim*n_ag,128)
        self.Linear2 = nn.Linear(128,1)

    def forward(self,x):
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x


class A3CNet(nn.Module):
    def __init__(self, s_dim, a_dim, CNN=False):
        super(A3CNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        if CNN:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=(1,1))
            self.pi1 = nn.Linear(1,32)
            self.pi2 = nn.Linear(32,32)
            self.LSTM = nn.LSTM(
                input_size=32,
                hidden_size=32,
                num_layers=1,
            )
        else:
            self.pi1 = nn.Linear(s_dim, 128)
            self.pi2 = nn.Linear(128, a_dim)
            self.v1 = nn.Linear(s_dim, 128)
            self.v2 = nn.Linear(128, 1)
            set_init([self.pi1, self.pi2, self.v1, self.v2])
            self.distribution = torch.distributions.Categorical


    def forward(self, x):
        pi1 = torch.relu(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.relu(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s, dist=False):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        if not dist:
            return m.sample().numpy()[0]
        else:
            return m.sample().numpy()[0], prob

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

class ActorRNN(nn.Module):
    def __init__(self,state_dim,action_dim,CNN=True):
        super(ActorRNN, self).__init__()
        self.CNN = CNN
        if CNN:
            self.Conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=(1,1))
            self.Linear_a = nn.Linear(action_dim, 32)
            self.Linear1 = nn.Linear(state_dim*2, 32)
            self.Linear2 = nn.Linear(32, 32)
            self.LSTM = nn.LSTM(
                input_size=32,
                hidden_size=128,
                num_layers=1,
            )
            self.out = nn.Linear(128,action_dim)
        else:
            self.rnn = nn.GRU(
                input_size=state_dim,
                hidden_size=128,
                num_layers=1,
            )
            self.out = nn.Linear(128,action_dim)

    def CNN_preprocess(self,x):
        x = self.Conv1(x)
        x = torch.relu(x)
        x = torch.flatten(x, start_dim=1,end_dim=-1).unsqueeze(0)
        return x

    def forward(self, x, a=None):
        # if self.CNN:
        #     x = self.Conv1(x)
        #     x = F.relu(x)
        #     x = torch.flatten(x,start_dim=1,end_dim=-1).unsqueeze(0)
        x = self.CNN_preprocess(x)
        x = torch.relu(self.Linear1(x))
        x = torch.relu(self.Linear2(x))
        x = x.reshape(-1,1,32)
        if not isinstance(a, type(None)):
            a = torch.relu(self.Linear_a(a))
            a = a.reshape(-1,1,32)
            x = x+a
        x, h_n = self.LSTM(x,None)
        x = F.relu(x[-1,:,:])
        x = self.out(x)
        return F.softmax(x)

class CriticRNN(nn.Module):
    def __init__(self,state_dim,action_dim,CNN=True):
        super(CriticRNN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.CNN = CNN
        if CNN:
            self.Conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=(1,1))
            self.Linear_a = nn.Linear(action_dim, 32)
            self.Linear1 = nn.Linear(state_dim*2, 32)
            self.Linear2 = nn.Linear(32, 32)
            self.LSTM = nn.LSTM(
                input_size=32,
                hidden_size=128,
                num_layers=1,
            )
            self.out = nn.Linear(128,1)
        else:
            self.rnn = nn.GRU(
                input_size=state_dim,
                hidden_size=128,
                num_layers=1,
            )
            self.out = nn.Linear(128,1)

    def CNN_preprocess(self, x):
        if not isinstance(x, type(torch.Tensor())):
            x = torch.Tensor(x)
        x = self.Conv1(x)
        x = torch.relu(x)
        x = torch.flatten(x, start_dim=1,end_dim=-1).unsqueeze(0)
        return x

    def forward(self, x, a=None):
        # if self.CNN:
        #     x = self.Conv1(x)
        #     x = F.relu(x)
        #     x = torch.flatten(x,start_dim=1,end_dim=-1).unsqueeze(0)
        x = self.CNN_preprocess(x)
        x = torch.relu(self.Linear1(x))
        x = torch.relu(self.Linear2(x))
        x = x.reshape(-1,1,32)
        if not isinstance(a, type(None)):
            a = torch.relu(self.Linear_a(a))
            x = x+a
        x, h_n = self.LSTM(x,None)
        x = F.relu(x[-1,:,:])
        x = self.out(x)
        return x

if __name__ == "__main__":
    model_name = "pg_social"
    file_name  = "train_para/"+model_name
    agentParam = {"ifload":True,"filename": file_name,"id":"0"}
    net = torch.load(agentParam["filename"]+"pg"+agentParam["id"]+".pth",map_location = torch.device('cuda'))
    optimizer = optim.Adam(net.parameters(), lr=0.01)
