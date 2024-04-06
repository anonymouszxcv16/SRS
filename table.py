import torch


def read_results(env_name, seed):
    with open(f"results/{env_name}/{baseline}_{seed}", "r") as file:
        lines = file.readlines()
        rewards_baseline = lines[0]

    with open(f"results/{env_name}/{competitor}_{seed}", "r") as file:
        lines = file.readlines()
        rewards_competitor = lines[0]

    rewards_baseline = rewards_baseline[1:-2].split(", ")
    rewards_competitor = rewards_competitor[1:-2].split(", ")

    rewards_baseline = [float(r) for r in rewards_baseline][:200]
    rewards_competitor = [float(r) for r in rewards_competitor][:200]

    return rewards_baseline, rewards_competitor


env_names = [
    "Ant-v2"
]

if __name__ == '__main__':

    baseline = "DDPG"
    competitor = "GFN"

    seeds = [0, 1, 2, 3]

    improvements = 0
    line = "Environment & DDPG & \\textit{GFlowNets} & Improvement \\\\ \n \\hline \n"

    for env_name in env_names:

        rewards_baseline, rewards_competitor = [], []

        for seed in seeds:
            reward_baseline, reward_competitor = read_results(env_name=env_name, seed=seed)

            rewards_baseline.append(reward_baseline)
            rewards_competitor.append(reward_competitor)

        rewards_baseline = torch.tensor(rewards_baseline)
        rewards_competitor = torch.tensor(rewards_competitor)

        competitor_max = rewards_competitor.mean(dim=0).max().item()
        baseline_max = rewards_baseline.mean(dim=0).max().item()

        improvement = ((competitor_max / baseline_max - 1) * 100)
        improvements += improvement

        line += f"{env_name} & " \
                f"{baseline_max:,.1f} & " \
                f"{competitor_max:,.1f} & " \
                f"{improvement:,.1f}\% \\\\ \n"

    print(line)
    print(f"& & & {improvements:,.1f}\% \\\\")