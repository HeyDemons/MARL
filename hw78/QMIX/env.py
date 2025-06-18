from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Callable, List, Tuple, Dict, Any


# --- 核心代码部分 ---

class PettingZooWrapper:
    """
    一个通用的 PettingZoo 环境封装器。
    
    它将 PettingZoo 环境返回的字典格式数据转换为列表格式，
    并自动处理智能体列表。
    """
    def __init__(self, env_fn: Callable[..., Any]):
        """
        初始化封装器。

        Args:
            env_fn (Callable[..., Any]): 一个用于创建 PettingZoo 环境的函数。
        """
        self.env = env_fn()
        # 智能体列表将在第一次 reset 后动态获取
        self.agents: List[str] = []

    @property
    def observation_space(self) -> Any:
        """获取单个智能体的观测空间。"""
        return self.env.observation_space(self.env.possible_agents[0])

    @property
    def action_space(self) -> Any:
        """获取单个智能体的动作空间。"""
        return self.env.action_space(self.env.possible_agents[0])

    def reset(self, seed: int = None) -> List[Any]:
        """
        重置环境。
        在第一次调用后，它会获取并存储智能体列表。
        """
        obs, infos = self.env.reset(seed=seed)
        self.agents = self.env.agents
        return self._dict_to_list(obs)

    def step(self, actions: List[Any]) -> Tuple[List[Any], List[float], List[bool], List[bool], List[Any]]:
        """在环境中执行一步。"""
        action_dict = self._list_to_dict(actions)
        obs, rewards, terminations, truncations, infos = self.env.step(action_dict)
        results = (obs, rewards, terminations, truncations, infos)
        return tuple(self._dict_to_list(d) for d in results)
    
    def render(self) -> Any:
        """渲染环境。"""
        return self.env.render()

    def close(self):
        """关闭环境。"""
        self.env.close()

    def _list_to_dict(self, data: List[Any]) -> Dict[str, Any]:
        """将列表数据转换为以智能体ID为键的字典。"""
        return {agent: data[i] for i, agent in enumerate(self.agents)}

    def _dict_to_list(self, data: Dict[str, Any]) -> List[Any]:
        """将以智能体ID为键的字典转换为列表。"""
        return [data[agent] for agent in self.agents]


# 修正点：将类型提示从 Pipe 改为更精确的 Connection
def worker(remote: Connection, parent_remote: Connection, env_fn: Callable[..., Any]):
    """
    工作进程函数，用于运行一个独立的环境实例。
    """
    parent_remote.close()
    env = PettingZooWrapper(env_fn)
    
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(env.step(data))
            elif cmd == 'reset':
                remote.send(env.reset(seed=data))
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError(f"未知命令: {cmd}")
    except EOFError:
        print("工作进程连接关闭。")
    finally:
        env.close()


class ParallelEnv:
    """
    一个管理多个并行环境实例的类。
    这个版本更加通用，可以接受任何创建环境的函数。
    """
    def __init__(self, env_fn: Callable[..., Any], n_envs: int):
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        
        self.processes = [
            Process(target=worker, args=(work_remote, remote, env_fn))
            for work_remote, remote in zip(self.work_remotes, self.remotes)
        ]

        for p in self.processes:
            p.daemon = True
            p.start()
        
        for remote in self.work_remotes:
            remote.close()
            
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step(self, actions: List[List[Any]]) -> Tuple[List[Any], List[Any], List[Any], List[Any], List[Any]]:
        """在所有并行环境中执行一步。"""
        for remote, action_list in zip(self.remotes, actions):
            remote.send(('step', action_list))
        
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, tructs, infos = zip(*results)
        return list(obs), list(rewards), list(dones), list(tructs), list(infos)

    def reset(self, seeds: List[int] = None) -> List[Any]:
        """重置所有并行环境。"""
        if seeds is None:
            seeds = [None] * self.n_envs
            
        for remote, seed in zip(self.remotes, seeds):
            remote.send(('reset', seed))
            
        return [remote.recv() for remote in self.remotes]

    def close(self):
        """关闭所有并行环境和工作进程。"""
        print("正在关闭所有并行环境...")
        for remote in self.remotes:
            remote.send(('close', None))
        
        for p in self.processes:
            p.join()
        print("所有环境已关闭。")