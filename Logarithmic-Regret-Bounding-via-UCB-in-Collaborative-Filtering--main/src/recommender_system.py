import numpy as np

from src.ucb_algorithm import UCB


def run_ucb_simulation(df, rounds=10000, seed=None):
    """
    Run a UCB bandit simulation where each movie is an arm.

    Notes:
    - The loop starts at t=1 to avoid log(0) in UCB.
    - Rewards are sampled from empirical rewards for the chosen movie.
    """
    rng = np.random.default_rng(seed)

    movies = df["movie_id"].unique()
    n_arms = len(movies)

    ucb = UCB(n_arms)

    rewards = []
    selections = []

    for t in range(1, rounds):
        arm = ucb.select_arm(t)
        movie = movies[arm]

        movie_rewards = df[df["movie_id"] == movie]["reward"].values
        reward = float(rng.choice(movie_rewards))

        ucb.update(arm, reward)

        rewards.append(reward)
        selections.append(int(arm))

    return {
        "rewards": np.array(rewards, dtype=float),
        "selections": np.array(selections, dtype=int),
        "movies": movies,
        "counts": ucb.counts.copy(),
        "values": ucb.values.copy(),
    }

