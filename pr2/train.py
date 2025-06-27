import os
import random
import time
from collections import deque

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers.normalize import RunningMeanStd

from config import Args
from models import Agent, RNDModel
from wrappers import RecordEpisodeStatistics
from utils import RewardForwardFilter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def train():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Initialize wandb if tracking
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    # Setup tensorboard
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Initialize environment
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
        repeat_action_probability=0.25,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Initialize models
    agent = Agent(envs).to(device)
    rnd_model = RNDModel(4, envs.single_action_space.n).to(device)
    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(
        combined_parameters,
        lr=args.learning_rate,
        eps=1e-5,
    )

    # Statistics tracking
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    # Storage initialization
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # Global variables
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    print("Start to initialize observation normalization parameter.....")
    next_ob = []
    for step in tqdm(range(args.num_steps * args.num_iterations_obs_norm_init), desc="Initializing observation normalization"):
        acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
        s, r, d, _ = envs.step(acs)
        next_ob += s[:, 3, :, :].reshape([-1, 1, 84, 84]).tolist()

        if len(next_ob) % (args.num_steps * args.num_envs) == 0:
            next_ob = np.stack(next_ob)
            obs_rms.update(next_ob)
            next_ob = []
    print("End to initialize...")

    # Main training loop
    pbar = tqdm(range(1, num_updates + 1), desc="Training Progress")
    for update in pbar:
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollout data
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                value_ext, value_int = agent.get_value(obs[step])
                ext_values[step], int_values[step] = (
                    value_ext.flatten(),
                    value_int.flatten(),
                )
                action, logprob, _, _, _ = agent.get_action_and_value(obs[step])

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            # Calculate intrinsic rewards with RND
            rnd_next_obs = (
                (
                    (next_obs[:, 3, :, :].reshape(args.num_envs, 1, 84, 84) - torch.from_numpy(obs_rms.mean).to(device))
                    / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                ).clip(-5, 5)
            ).float()
            target_next_feature = rnd_model.target(rnd_next_obs)
            predict_next_feature = rnd_model.predictor(rnd_next_obs)
            curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data
            
            # Log episode statistics
            for idx, d in enumerate(done):
                if d and info["lives"][idx] == 0:
                    avg_returns.append(info["r"][idx])
                    epi_ret = np.average(avg_returns)
                    print(
                        f"global_step={global_step}, episodic_return={info['r'][idx]}, curiosity_reward={np.mean(curiosity_rewards[step].cpu().numpy())}"
                    )
                    writer.add_scalar("charts/avg_episodic_return", epi_ret, global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar(
                        "charts/episode_curiosity_reward",
                        curiosity_rewards[step][idx],
                        global_step,
                    )
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # Calculate and normalize intrinsic rewards
        curiosity_reward_per_env = np.array(
            [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
        )
        mean, std, count = (
            np.mean(curiosity_reward_per_env),
            np.std(curiosity_reward_per_env),
            len(curiosity_reward_per_env),
        )
        reward_rms.update_from_moments(mean, std**2, count)

        curiosity_rewards /= np.sqrt(reward_rms.var)

        # Calculate advantages and returns
        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(curiosity_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            
            # GAE calculation
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                int_advantages[t] = int_lastgaelam = (
                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                )
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)

        # Combine intrinsic and extrinsic advantages
        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        # Update observation statistics
        obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy())

        # PPO update
        b_inds = np.arange(args.batch_size)

        # Prepare RND input
        rnd_next_obs = (
            (
                (b_obs[:, 3, :, :].reshape(-1, 1, 84, 84) - torch.from_numpy(obs_rms.mean).to(device))
                / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
            ).clip(-5, 5)
        ).float()

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # RND forward pass
                predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                forward_loss = F.mse_loss(
                    predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                ).mean(-1)

                # RND update with proportion masking
                mask = torch.rand(len(forward_loss), device=device)
                mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                )
                
                # Policy forward pass
                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # KL divergence calculation
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # Normalize advantages
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # PPO policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        combined_parameters,
                        args.max_grad_norm,
                    )
                optimizer.step()

            # Early stopping based on KL divergence
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # Logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/fwd_loss", forward_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        pbar.set_postfix(SPS=int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()