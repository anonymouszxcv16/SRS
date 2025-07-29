import argparse
import os
import time
import gym
import torch
import numpy as np
import random

import TD

# Train online RL agent.
def train_online(RL_agent, env, eval_env, args):
    # Performance.
    evals = []
    times = []

    # Initialize
    start_time = time.time()
    allow_train = False

    state, ep_finished = env.reset(), False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, times, t, start_time, args)

        # Select action.
        if allow_train:
            action = RL_agent.select_action(state)
        else:
            action = env.action_space.sample()

        # Do a step.
        state, reward, done, _ = env.step(action)

        if "Noise" in args.policy:
            sign = 1 if random.randint(0, 1) == 1 else -1
            noise = reward * random.random() * sign * RL_agent.args.noise_frac_max
            reward += noise

        ep_total_reward += reward
        ep_timesteps += 1
        done = float(1.) if done == 1 else 0

        # Store tuple.
        RL_agent.replay_buffer.add(state, action, state, reward, done)

        if allow_train:
            # Train.
            RL_agent.train()

        if done:
            if t >= args.timesteps_before_training:
                allow_train = True

            state, done = env.reset(), False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1


def train_offline(RL_agent, env, eval_env, args):
    # Performance.
    evals = []
    times = []

    # Collected Rewards Statistics.
    rewards_mean = []
    states_stds = []

    # Load offline dataset.
    RL_agent.replay_buffer.load_D4RL(d4rl.qlearning_dataset(env))
    start_time = time.time()

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, rewards_mean, states_stds, times, t, start_time, args, d4rl=True)

        # Train.
        RL_agent.train()

# Logs.
def maybe_evaluate_and_print(RL_agent, eval_env, evals, times, t, start_time, args, d4rl=False):
    if t % args.eval_freq == 0:
        # Total rewards.
        total_reward = np.zeros(args.eval_eps)

        for ep in range(args.eval_eps):
            state, done = eval_env.reset(), False
            step = 0

            # Episode
            while not done:
                # Action selection.
                action = RL_agent.select_action(np.array(state), use_exploration=False)

                # Step.
                state, reward, done, _ = eval_env.step(action)

                # Reward sum.
                total_reward[ep] += reward

                step += 1

        # Time.
        time_total = (time.time() - start_time) / 60

        # Score.
        score = eval_env.get_normalized_score(total_reward.mean()) * 100 if d4rl else total_reward.mean()

        print(f"Timesteps: {(t + 1):,.1f}\tMinutes {time_total:.1f}\tRewards: {score:,.1f}")

        # Performance
        evals.append(score)
        times.append(time_total)

        # file.
        with open(f"./results/{args.env}/{args.file_name}", "w") as file:
            file.write(f"{evals}\n{times}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Algorithm.
    parser.add_argument("--policy", default="DDPG", type=str)
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--noise_frac_max", default=.01, type=float)
    parser.add_argument('--use_checkpoints', default=True)

    # Exploration.
    parser.add_argument("--timesteps_before_training", default=2_500, type=int)
    parser.add_argument("--exploration_noise", default=.1, type=float)
    parser.add_argument("--discount", default=.99, type=float)
    parser.add_argument("--N", default=1, type=int)
    parser.add_argument("--buffer_size", default=int(1e6), type=int)

    # Q-target parameters.
    parser.add_argument("--alpha_sac", default=.2, type=float)

    # Environment.
    parser.add_argument("--env", default="Hopper-v2", type=str)
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

    # environment
    env = gym.make(args.env)
    env.action_space.seed(args.seed)

    eval_env = gym.make(args.env)
    eval_env.action_space.seed(args.seed)

    # Seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Environment
    obs_space = env.observation_space
    state_dim = int(np.prod(obs_space.shape))
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    RL_agent = TD.Agent(state_dim, action_dim, max_action, args)
    name = f"{args.policy}_{args.env}_{args.seed}"

    print("---------------------------------------")
    print(f"Algorithm: {args.policy}, Alpha: {args.alpha}, Buffer size: {args.buffer_size:,.1f}, "
          f"Environment: {args.env}, Noise Fraction Max: {args.noise_frac_max * 100:,.1f}%, Seed: {args.seed}, "
          f"Device: {RL_agent.device}")
    print("---------------------------------------")

    # Optimize.
    if args.offline == 1:
        train_offline(RL_agent, env, eval_env, args)
    else:
        train_online(RL_agent, env, eval_env, args)