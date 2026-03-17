import matplotlib.pyplot as plt


def make_cumulative_reward_fig(cumulative_rewards):
    fig, ax = plt.subplots()
    ax.plot(cumulative_rewards)
    ax.set_title("Cumulative Reward vs Rounds")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Cumulative reward")
    return fig


def make_cumulative_regret_fig(cumulative_regret):
    fig, ax = plt.subplots()
    ax.plot(cumulative_regret)
    ax.set_title("Cumulative Regret vs Rounds (optimal_reward = 1)")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Cumulative regret")
    return fig


def make_selection_hist_fig(selection_counts, bins=50):
    fig, ax = plt.subplots()
    ax.hist(selection_counts, bins=bins)
    ax.set_title("Distribution of Movie Selection Counts")
    ax.set_xlabel("Selections per movie (arm)")
    ax.set_ylabel("Number of movies")
    return fig


def save_figure(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")

