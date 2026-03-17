import streamlit as st

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.recommender_system import run_ucb_simulation
from src.evaluation import cumulative_reward, cumulative_regret, selection_counts
from src.visualization import (
    make_cumulative_regret_fig,
    make_cumulative_reward_fig,
    make_selection_hist_fig,
    save_figure,
)


st.title("UCB Movie Recommender System")

data = load_data("data/movielens_ucb_preprocessed.csv")

rounds = st.slider("Simulation rounds", min_value=100, max_value=20000, value=5000, step=100)

if st.button("Run Simulation"):
    out = run_ucb_simulation(data, rounds, seed=0)
    rewards = out["rewards"]
    selections = out["selections"]

    total_reward = float(np.sum(rewards))
    total_selections = int(selections.size)
    avg_reward = float(np.mean(rewards)) if total_selections else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total reward", f"{int(total_reward)}")
    m2.metric("Total selections", f"{total_selections}")
    m3.metric("Average reward", f"{avg_reward:.4f}")

    cum_reward = cumulative_reward(rewards)
    cum_regret = cumulative_regret(rewards, optimal_reward=1.0)
    counts = selection_counts(selections, n_arms=int(out["counts"].shape[0]))

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.subheader("Cumulative Reward")
    fig1 = make_cumulative_reward_fig(cum_reward)
    st.pyplot(fig1)

    st.subheader("Cumulative Regret")
    fig2 = make_cumulative_regret_fig(cum_regret)
    st.pyplot(fig2)

    st.subheader("Movie Selection Frequency (Histogram)")
    fig3 = make_selection_hist_fig(counts, bins=50)
    st.pyplot(fig3)

    # Save plots to project/results/
    reward_path = results_dir / f"ucb_cumulative_reward_rounds_{rounds}_{stamp}.png"
    regret_path = results_dir / f"ucb_cumulative_regret_rounds_{rounds}_{stamp}.png"
    hist_path = results_dir / f"ucb_selection_hist_rounds_{rounds}_{stamp}.png"

    save_figure(fig1, reward_path)
    save_figure(fig2, regret_path)
    save_figure(fig3, hist_path)

    st.success("Saved plots to results folder.")
    st.write("Saved files:")
    st.code(
        "\n".join([str(reward_path), str(regret_path), str(hist_path)]),
        language="text",
    )

