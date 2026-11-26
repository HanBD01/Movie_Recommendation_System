
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_profile(
    selected_genres,
    user_ratings,
    movie_features,
    available_genres,
    baseline=3.5
):
    """
    Use selected genres and rated movies to
    build a personalized genre profile.

    user_ratings:
        Dictionary mapping movie IDs to ratings: {movieId: rating}
    """

    profile = pd.Series(
        0.0,
        index=available_genres,
        dtype=float
    )

    # 1. Add selected genre preferences
    for genre in selected_genres:
        if genre in profile.index:
            profile[genre] += 1.0

    # 2. Incorporate the user's movie ratings
    movie_lookup = movie_features.set_index("movieId")

    for movie_id, rating in user_ratings.items():

        if movie_id not in movie_lookup.index:
            continue

        movie_genres = movie_lookup.loc[
            movie_id, available_genres
        ].astype(float)

        genre_count = movie_genres.sum()

        if genre_count == 0:
            continue

        preference = float(rating) - baseline

        profile += (
            movie_genres / genre_count
        ) * preference

    return profile


def recommend_content_movies(
    user_profile,
    movie_features,
    available_genres,
    watched_movie_ids=None,
    gamma=0.8,
    k=10
):
    """
    Combine genre similarity and normalized weighted ratings
    to recommend the top K movies.
    """

    if watched_movie_ids is None:
        watched_movie_ids = []

    # 1. Keep movies with at least one valid genre
    candidates = movie_features[
        movie_features[available_genres].sum(axis=1) > 0
    ].copy()

    # 2. Exclude movies already rated by the user
    candidates = candidates[
        ~candidates["movieId"].isin(watched_movie_ids)
    ].copy()

    if candidates.empty:
        return candidates

    # 3. Prepare user and movie genre vectors
    user_vector = (
        user_profile.reindex(available_genres)
        .fillna(0.0)
        .to_numpy(dtype=float)
        .reshape(1, -1)
    )

    movie_vectors = candidates[
        available_genres
    ].to_numpy(dtype=float)

    # 4. Calculate cosine similarity between genre vectors
    if np.linalg.norm(user_vector) > 0:
        similarities = cosine_similarity(
            user_vector,
            movie_vectors
        )[0]
    else:
        similarities = np.zeros(len(candidates))

    candidates["genre_similarity"] = similarities

    # 5. Calculate the original content-based score
    content_scores = (
        0.7 * similarities
        + 0.3 * candidates["normalized_rating"].to_numpy()
    )

    # 6. Calculate the popularity-aware score
    candidates["recommendation_score"] = (
        (1 - gamma) * content_scores
        + gamma * candidates["normalized_rating"].to_numpy()
    )

    # 7. Prioritize candidates with positive genre similarity
    positive_candidates = candidates[
        candidates["genre_similarity"] > 0
    ].copy()

    positive_candidates = positive_candidates.sort_values(
        "recommendation_score",
        ascending=False
    )

    recommendations = positive_candidates.head(k).copy()

    # 8. Use popularity fallback if fewer than K movies are recommended
    if len(recommendations) < k:

        remaining = candidates[
            ~candidates["movieId"].isin(
                recommendations["movieId"]
            )
        ].copy()

        remaining = remaining.sort_values(
            "normalized_rating",
            ascending=False
        )

        needed = k - len(recommendations)

        recommendations = pd.concat(
            [
                recommendations,
                remaining.head(needed)
            ],
            ignore_index=True
        )

    return recommendations.reset_index(drop=True)
