
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from .collaborative import calculate_cf_scores


def recommend_hybrid_movies(
    user_profile,
    user_ratings,
    movie_features,
    available_genres,
    user_item_matrix,
    similarity_cache=None,
    gamma=0.8,
    beta=0.2,
    k=10
):
    """
    Generate popularity-aware hybrid recommendations with fallback.

    user_profile:
        A Series of genre preference scores for the user.

    user_ratings:
        Dictionary mapping movie IDs to ratings: {movieId: rating}

    Returns:
        A DataFrame containing recommended movies.
    """

    # 1. Use only movies with valid genres as candidates
    candidates = movie_features[
        movie_features[available_genres].sum(axis=1) > 0
    ].copy()

    # 2. Exclude movies already rated by the user
    watched_ids = set(user_ratings.keys())

    candidates = candidates[
        ~candidates["movieId"].isin(watched_ids)
    ].copy()

    if candidates.empty:
        return candidates

    # 3. Prepare the user genre vector
    user_vector = (
        user_profile.reindex(available_genres)
        .fillna(0.0)
        .to_numpy(dtype=float)
        .reshape(1, -1)
    )

    # 4. Prepare movie genre vectors
    movie_vectors = candidates[
        available_genres
    ].to_numpy(dtype=float)

    # 5. Calculate genre cosine similarity
    if np.linalg.norm(user_vector) > 0:
        genre_similarities = cosine_similarity(
            user_vector,
            movie_vectors
        )[0]
    else:
        genre_similarities = np.zeros(
            len(candidates)
        )

    candidates["genre_similarity"] = genre_similarities

    # 6. Retrieve normalized weighted movie ratings
    popularity_scores = candidates[
        "normalized_rating"
    ].to_numpy(dtype=float)

    # 7. Calculate the original content-based score
    content_scores = (
        0.7 * genre_similarities
        + 0.3 * popularity_scores
    )

    # 8. Calculate the popularity-aware score
    popularity_aware_scores = (
        (1 - gamma) * content_scores
        + gamma * popularity_scores
    )

    # 9. Calculate collaborative filtering scores
    cf_scores = calculate_cf_scores(
        user_ratings=user_ratings,
        user_item_matrix=user_item_matrix,
        similarity_cache=similarity_cache
    )

    candidates["cf_score"] = (
        cf_scores.reindex(
            candidates["movieId"]
        ).to_numpy()
    )

    # Use a neutral value of zero for movies without CF scores
    cf_signal = (
        candidates["cf_score"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    # 10. Calculate the final hybrid score
    candidates["recommendation_score"] = np.clip(
        popularity_aware_scores
        + beta * (cf_signal / 1.5),
        0,
        1
    )

    # 11. Prioritize movies with positive genre similarity
    hybrid_candidates = candidates[
        candidates["genre_similarity"] > 0
    ].copy()

    hybrid_candidates = hybrid_candidates.sort_values(
        ["recommendation_score", "movieId"],
        ascending=[False, True]
    )

    recommendations = hybrid_candidates.head(k).copy()
    recommendations["recommendation_source"] = "Hybrid"

    # 12. Use popularity fallback if fewer than K movies are recommended
    if len(recommendations) < k:

        remaining = candidates[
            ~candidates["movieId"].isin(
                recommendations["movieId"]
            )
        ].copy()

        remaining = remaining.sort_values(
            ["normalized_rating", "movieId"],
            ascending=[False, True]
        )

        needed = k - len(recommendations)

        fallback_movies = remaining.head(needed).copy()

        fallback_movies["recommendation_source"] = (
            "Popularity Fallback"
        )

        recommendations = pd.concat(
            [recommendations, fallback_movies],
            ignore_index=True
        )

    return recommendations.reset_index(drop=True)
