import argparse
import os
import time

import gym
import pybullet_envs
import numpy as np
import torch

import TD


def train(RL_agent, env, eval_env, args):
    evals = []

    start_time = time.time()
    allow_train = False

    state, ep_finished = env.reset(), False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)

        if allow_train:
            action = RL_agent.select_action(np.array(state))
        else:
            action = env.action_space.sample()

        next_state, reward, ep_finished, _ = env.step(action)

        ep_total_reward += reward
        ep_timesteps += 1

        done = float(ep_finished) if ep_timesteps < env._max_episode_steps else 0
        RL_agent.replay_buffer.add(state, action, next_state, reward, done)

        state = next_state

        if allow_train:
            RL_agent.train()

        if ep_finished:

            if allow_train and args.use_checkpoints and "TD7" in args.policy:
                RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

            if t >= args.timesteps_before_training:
                allow_train = True

            state, done = env.reset(), False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1


def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args):
    if t % args.eval_freq == 0:
        total_reward = np.zeros(args.eval_eps)

        for ep in range(args.eval_eps):
            state, done = eval_env.reset(), False

            while not done:
                action = RL_agent.select_action(np.array(state), args.use_checkpoints, use_exploration=False)

                state, reward, done, _ = eval_env.step(action)
                total_reward[ep] += reward

        time_total = (time.time() - start_time) / 60
        score = total_reward.mean()

        print(f"Timesteps: {(t + 1):,.1f}\tMinutes {time_total:.1f}\tRewards: {score:,.1f}")

        evals.append(score)

        with open(f"./results/{args.env}/{args.file_name}", "w") as file:
            file.write(f"{evals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # RL
    parser.add_argument("--policy", default="ORL", type=str)
    parser.add_argument("--env", default="HumanoidStandup-v2", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--use_checkpoints', default=True)

    # Evaluation
    parser.add_argument("--timesteps_before_training", default=25_000, type=int)
    parser.add_argument("--eval_freq", default=5_000, type=int)
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--N", default=2, type=int)
    parser.add_argument("--buffer_size", default=1e6, type=int)

    # File
    parser.add_argument('--file_name', default=None)
    args = parser.parse_args()

    if args.file_name is None:
        args.file_name = f"{args.policy}_{args.seed}"

    if not os.path.exists(f"./results/{args.env}"):
        os.makedirs(f"./results/{args.env}")

    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    print("---------------------------------------")
    print(f"Algorithm: {args.policy}, Buffer size: {args.buffer_size:,.1f}, Environment: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    # Seed.
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    eval_env.seed(args.seed + 100)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    RL_agent = TD.Agent(state_dim, action_dim, max_action, args)
    name = f"{args.policy}_{args.env}_{args.seed}"

    # Optimize.
    train(RL_agent, env, eval_env, args)
