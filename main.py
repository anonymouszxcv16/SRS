import argparse
import os
import time
import gym
import torch
import numpy as np

import TD

# Train online RL agent.
def train_online(RL_agent, env, eval_env, args):
    # Performance.
    evals = []
    times = []
    alphas = []

    rewards_mean, rewards_std, biases, bc_losses, Ms = [], [], [], [], []

    # Initialize
    start_time = time.time()
    allow_train = False

    state, ep_finished = env.reset(), False
    trunc = False

    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    if "PointMaze" in args.env:
        state = state[0]['observation']

    # Gt
    t0 = 0

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        RL_agent.t += 1
        maybe_evaluate_and_print(RL_agent, eval_env, evals, times, alphas, rewards_mean, rewards_std, biases, bc_losses, Ms, t, start_time, args)

        # Select action.
        if allow_train:
            action = RL_agent.select_action(state)
        else:
            action = env.action_space.sample()

        # Do a step.
        if "PointMaze" in args.env:
            state_next, reward, done, trunc, _ = env.step(action)
        else:
            state_next, reward, done, _ = env.step(action)

        if "PointMaze" in args.env:
            state_next = state_next['observation']

        ep_total_reward += reward
        ep_timesteps += 1

        # done = float(ep_finished or ep_timesteps == env._max_episode_steps)
        done = float(ep_finished or trunc or ep_timesteps == env._max_episode_steps)

        # Store tuple.
        RL_agent.replay_buffer.add(torch.tensor(state).to(args.device), torch.tensor(action).to(args.device), torch.tensor(state_next).to(args.device), reward, done)

        state = state_next

        if allow_train:
            # Train.
            RL_agent.train()

        if done:
            if t >= args.timesteps_before_training:
                allow_train = True

            state, done = env.reset(), False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1

            if "PointMaze" in args.env:
                state = state[0]['observation']

            # Episode length
            T = RL_agent.t - t0
            scores = torch.zeros((T, 1)).to(args.device)

            # Gt
            for step in range(T):
                # t
                for t in range(step, T):
                    # r_t
                    r_t = RL_agent.replay_buffer.reward[RL_agent.t - t0 + t]

                    # Softplus Normalized
                    if "SRS" in args.policy or "RC" in args.policy:
                        r_avg = RL_agent.replay_buffer.reward[:RL_agent.replay_buffer.size].mean()
                        r_max = max(1, RL_agent.replay_buffer.reward[:RL_agent.replay_buffer.size].max())

                        r_t = (r_t - r_avg) / r_max

                        if "SRS" in args.policy:
                            r_t = (1 / args.alpha) * (1 + (args.alpha * (r_t)).exp()).log()

                    # r_t ** discount
                    scores[step] += r_t * args.discount ** (t - step)

            RL_agent.replay_buffer.mc_score[t0:RL_agent.t] = scores
            t0 = RL_agent.t


def train_offline(RL_agent, env, eval_env, args):
    # Performance.
    evals = []
    times = []
    alphas = []

    rewards_mean, rewards_std, biases, bc_losses, Ms = [], [], [], [], []

    # Load offline dataset.
    RL_agent.replay_buffer.load_D4RL(d4rl.qlearning_dataset(env))
    start_time = time.time()

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, times, alphas, rewards_mean, rewards_std, biases, bc_losses, Ms, t, start_time, args, d4rl=True)

        # Train.
        RL_agent.train()

# Logs.
def maybe_evaluate_and_print(RL_agent, eval_env, evals, times, alphas, rewards_mean, rewards_std, biases, bc_losses, Ms, t, start_time, args, d4rl=False):
    if t % args.eval_freq == 0:
        # Total rewards.
        q_values = np.zeros(args.eval_eps)
        discounted_reward = np.zeros(args.eval_eps)

        for ep in range(args.eval_eps):
            state, done = eval_env.reset(), False

            if "PointMaze" in args.env:
                state = state[0]['observation']

            action = RL_agent.select_action(state, deterministic=True)

            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float).to(RL_agent.args.device).unsqueeze(0)
                action = torch.tensor(action, dtype=torch.float).to(RL_agent.args.device).unsqueeze(0)

                Q_target = RL_agent.critic_target(state, action).min(1, keepdim=True)[0]

            q_values[ep] = Q_target

            step = 0

            # Episode
            while not done:
                # Action selection.
                action = RL_agent.select_action(state, use_exploration=False)

                # Step.
                if "PointMaze" in args.env:
                    state, reward, done, trunc, _ = eval_env.step(action)
                    done = done or trunc
                else:
                    state, reward, done, _ = eval_env.step(action)

                if "PointMaze" in args.env:
                    state = state['observation']

                # Reward sum.
                discounted_reward[ep] += reward * RL_agent.args.discount ** step
                
                step += 1

        # Time.
        time_total = (time.time() - start_time) / 60

        # Score.
        score = eval_env.get_normalized_score(discounted_reward.mean()) * 100 if d4rl else discounted_reward.mean()

        # Rewards distribution
        reward_mean = RL_agent.replay_buffer.reward[:RL_agent.replay_buffer.size].mean().item()
        reward_std = RL_agent.replay_buffer.reward[:RL_agent.replay_buffer.size].std().item()

        M = RL_agent.replay_buffer.reward.max().item()

        q_score = q_values.mean().item()
        bias = torch.tensor(score - q_score).abs().item()
        bc_loss = RL_agent.bc_loss / args.eval_freq
        RL_agent.bc_loss = 0

        print(f"Timesteps: {(t + 1):,.1f}\tMinutes {time_total:.1f}\tRewards: {score:,.1f}\t"
              f"Bias: {bias:,.2f}\t"
              f"BC loss: {bc_loss:.2f}\t"
              f"Mean(R): {reward_mean:,.2f}\t"
              f"Std(R): {reward_std:,.2f}\t"
              f"Alpha: {RL_agent.args.alpha:,.3f}\t"
              f"M: {M:.2f}\t")

        # Performance
        evals.append(score)
        times.append(time_total)
        alphas.append(RL_agent.args.alpha)

        rewards_mean.append(reward_mean)
        rewards_std.append(reward_std)
        bc_losses.append(bc_loss)
        biases.append(bias)

        Ms.append(M)

        # file.
        with open(f"./results/{args.env}/{args.file_name}", "w") as file:
            file.write(f"{evals}\n{times}\n{alphas}\n{rewards_mean}\n{rewards_std}\n{biases}\n{bc_losses}\n{Ms}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Algorithm.
    parser.add_argument("--policy", default="DDPG", type=str)
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--auto_alpha", default=0, type=int)
    parser.add_argument("--auto_alpha_interval", default=100_000, type=int)
    parser.add_argument("--noise_frac_max", default=.01, type=float)
    parser.add_argument('--use_checkpoints', default=True)

    # Exploration.
    parser.add_argument("--timesteps_before_training", default=25_000, type=int)
    parser.add_argument("--exploration_noise", default=.1, type=float)
    parser.add_argument("--discount", default=.99, type=float)
    parser.add_argument("--N", default=1, type=int)
    parser.add_argument("--buffer_size", default=int(1e6), type=int)

    # Q-target parameters.
    parser.add_argument("--alpha_sac", default=.2, type=float)

    # Environment.
    parser.add_argument("--env", default="Hopper-v2", type=str)
    parser.add_argument("--normalize", default=0, type=int)
    parser.add_argument("--offline", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)

    # Evaluation
    parser.add_argument("--eval_freq", default=5_000, type=int)
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)

    # File
    parser.add_argument('--file_name', default=None)
    args = parser.parse_args()

    if args.file_name is None:
        args.file_name = f"{args.policy}_{args.seed}"

    if not os.path.exists(f"./results/{args.env}"):
        os.makedirs(f"./results/{args.env}")

    # Offline.
    if args.offline == 1:
        import d4rl

        d4rl.set_dataset_path(args.d4rl_path)
        args.use_checkpoints = False

    else:
        import sys

        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]  # strip all args

        # import robust_gymnasium as gym  # safe import

        if "PointMaze" in args.env:
            import robust_gymnasium as gym  # safe import
            import gymnasium_robotics

            gym.register_envs(gymnasium_robotics)

    # environment
    env = gym.make(args.env)
    env.action_space.seed(args.seed)

    eval_env = gym.make(args.env)
    eval_env.action_space.seed(args.seed)

    # Seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if "PointMaze" in args.env:
        state, _ = env.reset()
        state_dim = state['observation'].shape[0]
    else:
        state_dim = env.observation_space.shape[0]

    # Environment
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    RL_agent = TD.Agent(state_dim, action_dim, max_action, args)
    name = f"{args.policy}_{args.env}_{args.seed}"

    print("---------------------------------------")
    print(f"Algorithm: {args.policy}, Alpha: {args.alpha}, Buffer size: {args.buffer_size:,.1f}, "
          f"Environment: {args.env}, Seed: {args.seed}, "
          f"Device: {RL_agent.device}")
    print("---------------------------------------")

    # Optimize.
    if args.offline == 1:
        train_offline(RL_agent, env, eval_env, args)
    else:
        train_online(RL_agent, env, eval_env, args)