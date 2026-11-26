
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

AVAILABLE_GENRES = [
    "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller",
    "War", "Western"
]


def load_movie_data():
    """Load movie metadata and rating data."""

    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")

    return movies, ratings


def build_movie_features(movies, ratings, m=10):
    """
    Build weighted rating and genre features for each movie.
    """

    # 1. Calculate per-movie rating statistics
    movie_stats = (
        ratings.groupby("movieId")["rating"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={
            "count": "rating_count",
            "mean": "rating_mean"
        })
    )

    # 2. Merge statistics with movie metadata
    features = movies.merge(
        movie_stats,
        on="movieId",
        how="left",
        validate="one_to_one"
    )

    # 3. Calculate the global mean rating
    global_mean = ratings["rating"].mean()

    features["rating_count"] = (
        features["rating_count"].fillna(0)
    )

    features["rating_mean"] = (
        features["rating_mean"].fillna(global_mean)
    )

    # 4. Weighted Rating
    v = features["rating_count"]
    R = features["rating_mean"]
    C = global_mean

    features["weighted_rating"] = (
        (v / (v + m)) * R
        + (m / (v + m)) * C
    )

    # 5. Apply min-max normalization
    min_rating = features["weighted_rating"].min()
    max_rating = features["weighted_rating"].max()

    if max_rating > min_rating:
        features["normalized_rating"] = (
            (features["weighted_rating"] - min_rating)
            / (max_rating - min_rating)
        )
    else:
        features["normalized_rating"] = 0.0

    # 6. One-hot encode movie genres
    genre_matrix = (
        features["genres"]
        .str.get_dummies(sep="|")
        .reindex(
            columns=AVAILABLE_GENRES,
            fill_value=0
        )
    )

    genre_matrix["movieId"] = features["movieId"].values

    # 7. Merge the final feature columns
    features = features.merge(
        genre_matrix,
        on="movieId",
        how="left",
        validate="one_to_one"
    )

    return features
