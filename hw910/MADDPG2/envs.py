import numpy as np
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from pettingzoo.mpe import simple_adversary_v3
from typing import Callable, List, Tuple, Dict, Any

# ## >> 修改点 1: 环境包装器被重构，使其更健壮和动态 <<
class AdversaryEnvWrapper:
    """
    一个为 PettingZoo 'simple_adversary_v3' 环境定制的健壮包装器。

    它能自动处理不同智能体之间观测维度的差异，并动态获取智能体列表。
    """
    def __init__(self, render_mode="human"):
        """
        初始化环境包装器。

        Args:
            render_mode (str): 环境的渲染模式。
        """
        self.env = simple_adversary_v3.parallel_env(render_mode=render_mode, continuous_actions=True)
        
        # 动态获取智能体列表，而不是硬编码
        self.agents = self.env.possible_agents
        self.num_agents = len(self.agents)
        
        # 动态计算并存储每个智能体所需的观测填充
        self._initialize_paddings()

    def _initialize_paddings(self):
        """
        检查所有智能体的观测空间，并计算使它们维度统一所需的填充。
        """
        obs_spaces = self.env.observation_spaces
        max_obs_dim = max(space.shape[0] for space in obs_spaces.values())
        
        self.paddings = {}
        for agent_id, space in obs_spaces.items():
            padding_size = max_obs_dim - space.shape[0]
            if padding_size > 0:
                self.paddings[agent_id] = np.zeros(padding_size)
            else:
                self.paddings[agent_id] = None

    def _process_obs(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """使用预先计算的填充来处理观测字典。"""
        for agent_id, obs in obs_dict.items():
            if self.paddings[agent_id] is not None:
                obs_dict[agent_id] = np.concatenate([obs, self.paddings[agent_id]])
        return obs_dict

    def reset(self, seed: int = None) -> Tuple[List[np.ndarray], List[Dict]]:
        """重置环境并返回列表格式的观测和信息。"""
        obs, infos = self.env.reset(seed=seed)
        processed_obs = self._process_obs(obs)
        return self._dict_to_list(processed_obs), self._dict_to_list(infos)

    def step(self, actions: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[Dict]]:
        """
        在环境中执行一步，并将连续动作转换为离散力。
        
        Args:
            actions (np.ndarray): 形状为 (num_agents, 2) 的动作数组, 值在 [-1, 1]。
        """
        # 将 [-1, 1] 的动作转换为环境所需的5维力向量
        # [no_op, move_left, move_right, move_down, move_up]
        full_actions = np.zeros((self.num_agents, 5), dtype=np.float32)
        full_actions[:, 1] = -actions[:, 0]  # 向左的力
        full_actions[:, 2] = actions[:, 0]   # 向右的力
        full_actions[:, 3] = -actions[:, 1]  # 向下的力
        full_actions[:, 4] = actions[:, 1]   # 向上的力
        
        # 力必须是非负的。 .clip(min=0) 是比 np.where 更高效、更惯用的写法
        full_actions = full_actions.clip(min=0)
        
        # 将动作列表转换为环境期望的字典格式
        action_dict = self._list_to_dict(full_actions)
        
        obs, rewards, dones, truncs, infos = self.env.step(action_dict)
        processed_obs = self._process_obs(obs)
        
        # 将结果从字典转换回列表
        results = (processed_obs, rewards, dones, truncs, infos)
        return tuple(self._dict_to_list(d) for d in results)

    def _list_to_dict(self, data_list: List[Any]) -> Dict[str, Any]:
        """使用动态智能体列表将数据从列表转换为字典。"""
        return {agent: data for agent, data in zip(self.agents, data_list)}

    def _dict_to_list(self, data_dict: Dict[str, Any]) -> List[Any]:
        """使用动态智能体列表将数据从字典转换为列表。"""
        return [data_dict[agent] for agent in self.agents]
    
    def render(self):
        """渲染环境。"""
        return self.env.render()

    def close(self):
        """关闭环境。"""
        self.env.close()

# ## >> 修改点 2: 使用与之前一致的、通用的 Worker 和 ParallelEnv <<
def worker(remote: Connection, parent_remote: Connection, env_fn: Callable[..., Any]):
    """多处理工作者的目标函数。"""
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(env.step(data))
            elif cmd == 'reset':
                remote.send(env.reset())
            elif cmd == 'close':
                break
            else:
                raise NotImplementedError(f"未知命令: {cmd}")
    except EOFError:
        pass
    finally:
        env.close()

class ParallelEnv:
    """
    一个通用的、使用多处理并行运行多个环境实例的类。
    """
    def __init__(self, env_fn: Callable[..., Any], n_envs: int):
        """
        初始化并行环境。

        Args:
            env_fn (Callable[..., Any]): 一个返回环境实例的函数。
            n_envs (int): 要并行运行的环境数量。
        """
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        
        self.processes = [Process(target=worker, args=(work_remote, remote, env_fn))
                          for work_remote, remote in zip(self.work_remotes, self.remotes)]
        
        for p in self.processes:
            p.daemon = True
            p.start()
        
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """对所有并行环境执行一步。"""
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, truncs, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(dones), np.array(truncs), list(infos)

    def reset(self, seeds: List[int] = None) -> np.ndarray:
        """重置所有并行环境。"""
        # Note: seeds are not implemented in this worker's reset for simplicity
        for remote in self.remotes:
            remote.send(('reset', None))
        
        results = [remote.recv() for remote in self.remotes]
        obs, _ = zip(*results)
        return np.array(obs)

    def close(self):
        """关闭所有并行环境。"""
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except BrokenPipeError:
                pass
        for p in self.processes:
            p.join()