from dataclasses import dataclass
from typing import Optional

@dataclass
class Args:
    """PPO+RND arguments"""
    # General parameters
    env_id: str = "MontezumaRevenge-v5"
    track: bool = False
    wandb_project_name: str = "ppo-rnd-montezuma"
    wandb_entity: Optional[str] = None
    cuda: bool = True
    torch_deterministic: bool = True
    seed: int = 1
    exp_name: str = "ppo-rnd"
    
    # PPO parameters
    learning_rate: float = 1e-4
    total_timesteps: int =  2_000_000_000
    num_envs: int = 128
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.999
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    
    # RND parameters
    int_coef: float = 1.0
    ext_coef: float = 2.0
    int_gamma: float = 0.99
    update_proportion: float = 0.25
    num_iterations_obs_norm_init: int = 50
    
    # Computed during runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0