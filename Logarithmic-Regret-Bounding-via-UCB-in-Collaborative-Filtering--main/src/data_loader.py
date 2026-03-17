import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def get_movies(df):
    return df["movie_id"].unique()

