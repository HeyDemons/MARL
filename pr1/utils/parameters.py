import argparse
def parameters():
    parser = argparse.ArgumentParser()
    ############################################ 选择环境 ############################################
    parser.add_argument("--env_name", type =str, default = "simple_tag_v3", help = "name of the env",   
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3']) 
    parser.add_argument("--algorithm", type = str, default = "MADDPG", help = "name of the algorithm",
                        choices=['MADDPG', 'QMIX', 'MASAC ','MAPPO'])
    parser.add_argument("--render_mode", type=str, default = "None", help = "None | human | rgb_array")
    parser.add_argument("--seed", type = int, default = 100, help = "random seed")
    parser.add_argument("--episode_num", type = int, default = 50000) # 5000
    parser.add_argument("--episode_length", type = int, default = 25) #50
    parser.add_argument('--learn_interval', type=int, default=1,
                        help='steps interval between learning time')
    parser.add_argument('--update_interval', type=int, default=1,
                        help='steps interval between update target network')
    parser.add_argument('--random_steps', type=int, default=2e4, help='random steps before the agent start to learn') #  2e3
    parser.add_argument('--tau', type=float, default=0.01, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=256, help='batch-size of replay buffer')  
    parser.add_argument('--actor_lr', type=float, default=5e-4, help='learning rate of actor') # .00002
    parser.add_argument('--critic_lr', type=float, default=5e-4, help='learning rate of critic') # .002
    parser.add_argument("--alpha_lr", type = float, default = 2e-4, help = "learning rate of alpha") # 0.01
    # The parameters for the communication network
    parser.add_argument('--size_win', type=int, default=200, help='visdom window size')
    # DDPG 独有参数 
    ## gauss noise
    parser.add_argument("--gauss_sigma", type=float, default=1) # 高斯标准差 # 0.1 
    parser.add_argument("--gauss_scale", type=float, default=1)
    parser.add_argument("--gauss_init_scale", type=float, default=1) # 若不设置衰减，则设置成None 若设置了衰减 则 sigma = gauss_scale * gauss_sigma
    parser.add_argument("--gauss_final_scale", type=float, default=0)
    ## OU noise
    parser.add_argument("--ou_sigma", type=float, default=1) # 
    parser.add_argument("--ou_dt", type=float, default=1)
    parser.add_argument("--init_scale", type=float, default=1) # 若不设置衰减，则设置成None # maddpg值:ou_sigma 0.2 init_scale:0.3
    parser.add_argument("--final_scale", type=float, default=0.0) 
    # trick参数
    parser.add_argument("--supplement", type=dict, default={'weight_decay':True,'OUNoise':True,'ObsNorm':False,'net_init':True,'Batch_ObsNorm':True})
    args = parser.parse_args()
    return args