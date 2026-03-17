# UCB Collaborative Filtering (MovieLens 100K)

University project for **Numerical Optimization**: *Logarithmic Regret Bounding via UCB in Collaborative Filtering*.

This project treats each **movie as an arm** in a multi-armed bandit and uses **UCB** to balance exploration vs exploitation.

## Project structure

```text
project/
  data/
    movielens_ucb_preprocessed.csv
  src/
    data_loader.py
    ucb_algorithm.py
    recommender_system.py
    evaluation.py
    visualization.py
  app/
    streamlit_app.py
  notebooks/
    experiment.ipynb
  requirements.txt
```

## Setup

From the `project/` directory:

```bash
pip install -r requirements.txt
```

## Dataset

Place your preprocessed file at:

`project/data/movielens_ucb_preprocessed.csv`

If your file currently sits in the workspace root (next to `project/`), the loader will also auto-detect it.

## Run Streamlit demo

From the `project/` directory:

```bash
streamlit run app/streamlit_app.py
```

## Run a quick simulation (no UI)

```bash
python -m src.evaluation
```

