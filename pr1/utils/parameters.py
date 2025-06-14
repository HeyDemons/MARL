import argparse
def parameters():
    parser = argparse.ArgumentParser()
    ############################################ 选择环境 ############################################
    parser.add_argument("--env_name", type =str, default = "simple_tag_v3", help = "name of the env",   
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3']) 
    parser.add_argument("--algorithm", type = str, default = "MASAC", help = "name of the algorithm",
                        choices=['MADDPG', 'QMIX', 'MASAC ','MAPPO'])
    parser.add_argument("--alpha_lr", type = float, default = 2e-4, help = "learning rate of alpha") # 0.01
    parser.add_argument("--render_mode", type=str, default = "None", help = "None | human | rgb_array")
    parser.add_argument("--episode_num", type = int, default = 2000) # 5000
    parser.add_argument("--episode_length", type = int, default = 200) #50
    parser.add_argument('--learn_interval', type=int, default=10,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=3e4, help='random steps before the agent start to learn') #  2e3
    parser.add_argument('--tau', type=float, default=0.005, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=512, help='batch-size of replay buffer')  
    parser.add_argument('--actor_lr', type=float, default=3e-4, help='learning rate of actor') # .00002
    parser.add_argument('--critic_lr', type=float, default=3e-3, help='learning rate of critic') # .002
    # The parameters for the communication network
    parser.add_argument('--visdom', type=bool, default=False, help='whether to use visdom for visualization')
    parser.add_argument('--size_win', type=int, default=100, help='visdom window size')
    args = parser.parse_args()
    return args