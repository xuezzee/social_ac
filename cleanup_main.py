import argparse
import torch
import numpy as np
import torch.multiprocessing as mp
from envs.SocialDilemmaENV.social_dilemmas.envir.cleanup import CleanupEnv
from envs.SocialDilemmaENV.social_dilemmas.envir.harvest import HarvestEnv
from PGagent import A3C, SocialInfluence, IAC_RNN, influence_A3C
from parallel_env_process import envs_dealer
from MAAC.algorithms.attention_sac import AttentionSAC
from MAAC.utils.buffer import ReplayBuffer
from utils import env_wrapper, make_parallel_env, Logger, Runner, create_seq_obs, plot
# from logger import Logger
from network import A3CNet, A3CAgent
from ddpg import DDPG
from torch.utils.tensorboard import SummaryWriter

# from envs.ElevatorENV import Lift

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=1, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=True, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs0 (default: 10)')
args = parser.parse_args()

# env = GatheringEnv(2)  # gym.make('CartPole-v1')
# env.seed(args.seed)
# torch.manual_seed(args.seed)

agentParam = {"gamma": args.gamma, "LR": 1e-2, "device": device,"ifload":False,"filename": None}
# agentParam =

model_name = "pg_social"
file_name = "/Users/xue/Desktop/Social_Law/saved_weight/" + model_name
save_eps = 10
ifsave_model = True
logger = Logger('./logs3')


def A3C_main_multiProcess():
    n_agents = 5
    first_decision_agent = 1
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    n_workers = mp.cpu_count()
    n_workers = 2
    messager = [mp.Pipe() for i in range(n_workers)]
    sender = [m[0] for m in messager]
    receiver = [m[1] for m in messager]
    glaw = DDPG([15,15,3], 8, seq_len=4, device=device)
    global_net = [A3CNet(675*2, 8, device=device).to(device)]
    global_net = global_net + [A3CNet(675*2, 8, device=device).to(device) for i in range(n_agents)]
    optimizer = [torch.optim.Adam(global_net[i].parameters(), lr=0.001) for i in range(n_agents)]
    scheduler_lr = [torch.optim.lr_scheduler.StepLR(optimizer[i],step_size=2000, gamma=0.99, last_epoch=-1) for i in range(n_agents)]
    envs = [env_wrapper(HarvestEnv(num_agents=n_agents),flatten=False) for i in range(n_workers)]
    workers = [SocialInfluence(
                   envs[worker],
                   global_net,
                   optimizer,
                   global_ep,
                   global_ep_r,
                   res_queue,
                   worker,
                   675, 8,
                   n_agents,
                   scheduler_lr,
                   # take_action_inTurn=True,
                   # first_decision_num=first_decision_agent,
                   glaw=glaw
                   ) for worker in range(n_workers)]
    for i in range(n_workers):
        workers[i].sender = sender[i]
    for worker in workers:
        worker.start()
    res = []
    while True:
        msg = [rec.recv() for rec in receiver]
        plot(msg, logger)
        # r = res_queue.get()
        # if r is not None:
        #     res.append(r)
        # else:
        #     break
    [w.join() for w in workers]

def A3C_main():
    n_agents = 5
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    n_workers = mp.cpu_count()
    n_workers = 3
    sender, recevier = mp.Pipe()
    global_net = [A3CNet(675, 9).to(device)]
    global_net = global_net + [A3CNet(676, 9).to(device) for i in range(n_agents-1)]
    optimizer = [torch.optim.Adam(global_net[i].parameters(), lr=0.001) for i in range(n_agents)]
    scheduler_lr = [torch.optim.lr_scheduler.StepLR(optimizer[i],step_size=1000000, gamma=0.9, last_epoch=-1) for i in range(n_agents)]
    envs = [env_wrapper(CleanupEnv(num_agents=n_agents),flatten=True) for i in range(n_workers)]
    # workers = [A3C(envs[worker], global_net, optimizer, global_ep, global_ep_r, res_queue, worker, 675, 9, n_agents, scheduler_lr) for worker in range(n_workers)]
    workers = [SocialInfluence(envs[worker], global_net, optimizer, global_ep, global_ep_r, res_queue, worker, 675, 9, n_agents,
                   scheduler_lr, device) for worker in range(n_workers)]
    workers[0].sender = sender
    for worker in workers:
        worker.start()
    res = []
    while True:
        msg = recevier.recv()
        # logger.scalar_summary("reward", msg[0], msg[1])
        # r = res_queue.get()
        # if r is not None:
        #     res.append(r)
        # else:
        #     break
    [w.join() for w in workers]

def A2C_main():
    n_agent = 5
    env = CleanupEnv(num_agents=n_agent)
    # action_dim = env.action_space
    # state_dim = env.observation_space
    agents = [IAC_RNN(9, 675, agentParam, useLaw=False, useCenCritc=False, num_agent=n_agent, device=device,
                      width=15, height=15, channel=3, name="agent%d"%i) for i in range(n_agent)]
    runner = Runner(env, n_agent, agents, logger=None)
    runner.run()

def A3C_SocialInfluence_main(): #新的parallel主函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    n_agents = 5
    n_influencer = 1
    n_influencee = 4
    n_rollout_threads = 2
    seq_len = 5
    steps_per_ep = 1000
    seq_obs = np.zeros((n_agents, seq_len, n_rollout_threads, 3, 15, 15))
    seq_act = np.zeros((n_agents, seq_len, n_rollout_threads, 9))
    env = make_parallel_env(n_rollout_threads, [CleanupEnv, n_agents], flatten=False)  #创建并行环境，flatten=False说明返回的是RGB图像（已经转换为(3,15,15)）
    agent_init_params = [{'num_in_pol': 675, 'num_out_pol': 9} for i in range(n_agents)]  #暂时没啥用
    influencer = [A3CAgent(9, 15, 15, influencer=True, device=device).to(device) for i in range(n_influencer)]  #influencer, influencee共5个
    influencee = [A3CAgent(9, 15, 15, influencer=False, device=device).to(device) for i in range(n_influencee)]
    agents = influence_A3C(obs_dim=675,                                                 #集中处理update和choose_action
                           act_dim=9,
                           lr=0.001,
                           agents=influencer+influencee,
                           width=15,
                           height=15,
                           channel=3,
                           lr_scheduler=True,
                           device=device)
    replay_buffer = ReplayBuffer(max_steps=10000,                                         #replayBuffer,和原来的MAAC类似，但是observation改成了RGB结构，里面多加了新的内容，主要用于influence
                                 num_agents=n_agents,                                     #还没debug，不知道好用不，详见buffer
                                 obs_dims=[675 for i in range(n_agents)],
                                 ac_dims=[9 for i in range(n_agents)],
                                 width=[15 for i in range(n_agents)],
                                 height=[15 for i in range(n_agents)],
                                 channel=[3 for i in range(n_agents)])
    for i_episode in range(101):
        n_state, ep_reward1, ep_reward2 = env.reset(), 0, 0
        for step in range(steps_per_ep):                                                                                            #！！！！以下部分还未debug，应该跑不了
            print(step)
            seq_obs = create_seq_obs(seq_obs, n_state.transpose(2,1,0,3,4,5), l=seq_len)
            torch_obs = [torch.autograd.Variable(torch.Tensor(np.vstack(seq_obs[:, i])).to(device),requires_grad=False) for i in range(n_agents)]  #转换obs格式，MAAC原句
            influencer_onehot, influencer_prob, influencer_int, influencer_logists = agents.choose_influencer_action(torch_obs)    #选择influencer动作，包括int logist onehot prob格式
            act_input = np.array([influencer_onehot[0] for i in range(n_agents)])[np.newaxis,:,:,:]
            seq_act = create_seq_obs(seq_act, act_input, l=seq_len)
            torch_act = [torch.autograd.Variable(torch.Tensor(np.vstack(seq_act[:, i])).to(device),requires_grad=False) for i in range(n_agents)]
            influencee_onehot, influencee_logist = agents.choose_action(torch_obs, torch_act)                        #利用obs和influencer onehot action 选择influencee动作
            torch_agent_actions = influencer_onehot + [influencee_onehot[i].data.cpu().numpy() for i in range(len(influencee))]
            actions_logists = influencer_logists + influencee_logist
            influencer_action = [[influencer_onehot[0] for i in range(n_agents)]for i in range(n_rollout_threads)]           #influencer的onehot action，将来计算counterfactual action使用

            agent_actions = torch_agent_actions                                          #将action转换为可以送入环境的格式
            actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]
            # influencer_action = [[agent_actions[0] for ac in n_agents] for i in range(n_rollout_threads)]
            n_state_, n_reward, done, _ = env.step(actions)                                                          #并行采样，细节见parallel_env_process
            ep_reward1 += sum(n_reward[0]); ep_reward2 += sum(n_reward[1])
            seq_obs_ = create_seq_obs(seq_obs, n_state_.transpose(2,1,0,3,4,5), seq_len)
            actions_logists = np.array([act_log.data.cpu().numpy() for act_log in actions_logists])
            replay_buffer.push(seq_obs.transpose(2,1,0,3,4,5), agent_actions, n_reward, seq_obs_.transpose(2,1,0,3,4,5), done, actions_logists, seq_act.transpose(2,1,0,3))    #样本送入replay_buffer
            step += n_rollout_threads                                                                                   #每次相当于走n_rollout_threads个step？？不确定
            if (len(replay_buffer) >= 1024 and (step % 100) < 12):
                if device == torch.device("cpu"):
                    use_gpu = False
                else:
                    use_gpu = True
                for u_i in range(4):
                    sample = replay_buffer.sample(1024,                                                                 #采样更新，还没有加时间序列
                                                  to_gpu=use_gpu)
                    agents.update(sample)
                # agents.prep_rollouts(device='cpu')
            n_state = n_state_
        print(ep_reward1,ep_reward2)


if __name__ == '__main__':
    mp.set_start_method("spawn", True)
    A3C_main_multiProcess()
    # A2C_main()
    # A3C_SocialInfluence_main()