import numpy as np


def cumulative_reward(rewards):
    return np.cumsum(rewards)


def cumulative_regret(rewards, optimal_reward=1.0):
    """
    Cumulative regret under the assumption that the optimal expected reward is 1.

    regret_t = optimal_reward - reward_t
    cumulative_regret_t = sum_{s=1..t} regret_s
    """
    rewards = np.asarray(rewards, dtype=float)
    regret_t = optimal_reward - rewards
    return np.cumsum(regret_t)


def selection_counts(selections, n_arms=None):
    """
    Count how often each arm was selected.
    Returns an array of length n_arms (or max_arm+1 if not provided).
    """
    selections = np.asarray(selections, dtype=int)
    if selections.size == 0:
        if n_arms is None:
            return np.array([], dtype=int)
        return np.zeros(int(n_arms), dtype=int)

    if n_arms is None:
        n_arms = int(selections.max()) + 1
    counts = np.bincount(selections, minlength=int(n_arms))
    return counts


if __name__ == "__main__":
    # Small smoke run so you can execute:
    #   python -m src.evaluation
    from src.data_loader import load_data
    from src.recommender_system import run_ucb_simulation

    df = load_data("data/movielens_ucb_preprocessed.csv")
    out = run_ucb_simulation(df, rounds=2000, seed=0)
    rewards = out["rewards"]
    print("Total Reward:", int(np.sum(rewards)))

