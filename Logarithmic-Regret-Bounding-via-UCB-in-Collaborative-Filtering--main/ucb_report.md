# 🎬 Logarithmic Regret Bounding via UCB in Collaborative Filtering

## 📌 Abstract
Cold-start problems in recommender systems require balancing exploration (trying new items) and exploitation (using known preferences). This project reformulates collaborative filtering as a Multi-Armed Bandit (MAB) problem and implements an Upper Confidence Bound (UCB) algorithm integrated with an Online Convex Optimization (OCO) framework. The proposed method dynamically adapts exploration using subgradient-based variance bounds and achieves logarithmic regret growth over time.

## 🧠 Introduction
Traditional recommender systems struggle with:
- New users (no history)
- New items (no ratings)

We model each movie as an arm and use UCB for decision-making.

## ⚙️ Methodology
UCB formula:
A_t = argmax (mu_i + sqrt(2 ln t / n_i))

## 📊 Experimental Setup
- Dataset: MovieLens
- Rounds: 5000

## 📈 Results
- Increasing cumulative reward
- Logarithmic regret growth
- Efficient arm selection

## 🚀 Conclusion
UCB with OCO provides faster convergence and better cold-start handling.
