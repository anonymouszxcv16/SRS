import matplotlib.pyplot as plt
import torch

# Plot rewards.
def plot_graph(xs,
               results_baseline,
               results_competitor,
               name,
               xlabel="Timesteps",
               ylabel="Rewards",
               fontsize=20):
    plt.rc('legend', fontsize=fontsize)
    fig, ax = plt.subplots()

    # baseline
    stderr_baseline = torch.std(results_baseline, dim=0) / torch.sqrt(torch.tensor(results_baseline.shape[0]))
    ax.plot(xs, torch.mean(results_baseline, dim=0), label=f"{baseline}", color=color_b)
    ax.fill_between(x=xs, y1=torch.mean(results_baseline, dim=0) - stderr_baseline,
                    y2=torch.mean(results_baseline, dim=0) + stderr_baseline,
                    color=color_b, alpha=.2)

    # competitor.
    stderr_competitor = torch.std(results_competitor, dim=0) / torch.sqrt(torch.tensor(results_competitor.shape[0]))
    ax.plot(xs, torch.mean(results_competitor, dim=0), label=f"{competitor_label}", color=color_c)
    ax.fill_between(x=xs, y1=torch.mean(results_competitor, dim=0) - stderr_competitor,
                    y2=torch.mean(results_competitor, dim=0) + stderr_competitor,
                    color=color_c, alpha=.2)

    # axises.
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    plt.legend()

    # axises.
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    plt.savefig(f"./results/{name}.png")


# Read results for all plots and tables.
def read_results(env_name, seed, axis=0):
    with open(f"results/{env_name}/{baseline}_{seed}", "r") as file:
        lines = file.readlines()
        results_baseline = lines[0]

    with open(f"results/{env_name}/{competitor}_{seed}", "r") as file:
        lines = file.readlines()
        results_competitor = lines[axis]

    results_baseline = [float(r) for r in results_baseline[1:-2].split(", ")]
    results_competitor = [float(r) for r in results_competitor[1:-2].split(", ")]

    return results_baseline, results_competitor


if __name__ == '__main__':
    # Baseline and competitors for plots.
    baseline = "DDPG"
    competitor = "GFN"

    color_b = "silver"
    color_c = "gold"

    # Env names for plots.
    env_name = "Ant-v2"

    alpha = 1
    competitor_label = f"GFlowNets"
    axis = 0

    # total points for plots.
    steps_tot = 1_000_000
    eval_freq = 5_000
    pts_tot = steps_tot // eval_freq
    xs = torch.arange(0, steps_tot, eval_freq)

    seeds = [0, 1, 2, 3]

    title = env_name.split("-")[0]
    name = f"{title}_{len(seeds)}"

    # read results.
    results_baseline, results_competitor = [], []

    # combine seeds.
    for seed in seeds:
        result_baseline, result_competitor = read_results(env_name=env_name, seed=seed, axis=axis)

        results_baseline.append(result_baseline[:pts_tot])
        results_competitor.append(result_competitor[:pts_tot])

    # tensors.
    results_baseline = torch.tensor(results_baseline)
    results_competitor = torch.tensor(results_competitor)

    # plot rewards.
    plot_graph(xs=xs,
               results_baseline=results_baseline,
               results_competitor=results_competitor,
               name=name,
               ylabel="Rewards")