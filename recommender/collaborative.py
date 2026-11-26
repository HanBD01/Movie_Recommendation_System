
import numpy as np
import pandas as pd


def build_user_item_matrix(ratings):
    """Create a user-item rating matrix from MovieLens ratings."""

    return ratings.pivot(
        index="userId",
        columns="movieId",
        values="rating"
    )



def calculate_item_similarities(
    movie_id,
    user_item_matrix,
    min_common_users=10,
    shrinkage=20
):
    """
    Calculate Pearson similarity using users who rated both movies.
    """

    movie_ids = user_item_matrix.columns

    if movie_id not in movie_ids:
        return pd.Series(0.0, index=movie_ids)

    target = user_item_matrix[movie_id]

    # 1. Count users who rated both movies
    common_counts = (
        user_item_matrix.notna()
        .mul(target.notna(), axis=0)
        .sum(axis=0)
    )

    # 2. Calculate Pearson correlation using common raters
    pearson = user_item_matrix.corrwith(
        target,
        axis=0
    ).fillna(0.0)

    # 3. Apply shrinkage
    shrinkage_weights = (
        common_counts /
        (common_counts + shrinkage)
    )

    similarities = pearson * shrinkage_weights

    # 4. Require a minimum number of common raters
    similarities[common_counts < min_common_users] = 0.0

    # 5. Exclude the target movie itself
    similarities.loc[movie_id] = 0.0

    return similarities.astype(float)


def calculate_cf_scores(
    user_ratings,
    user_item_matrix,
    similarity_cache=None,
    baseline=3.5,
    min_support=2,
    denominator_smoothing=0.5
):
    """
    Calculate collaborative filtering scores from a new user's movie ratings.

    user_ratings:
        {movieId: rating}

    Returns:
        A Series of CF scores indexed by movie ID.
    """

    if similarity_cache is None:
        similarity_cache = {}

    movie_ids = user_item_matrix.columns

    numerator = pd.Series(
        0.0,
        index=movie_ids
    )

    denominator = pd.Series(
        0.0,
        index=movie_ids
    )

    support = pd.Series(
        0,
        index=movie_ids
    )

    for movie_id, rating in user_ratings.items():

        if movie_id not in movie_ids:
            continue

        if movie_id not in similarity_cache:
            similarity_cache[movie_id] = (
                calculate_item_similarities(
                    movie_id,
                    user_item_matrix
                )
            )

        similarities = similarity_cache[movie_id]

        positive_similarities = similarities.clip(
            lower=0
        )

        preference = float(rating) - baseline

        numerator += (
            positive_similarities * preference
        )

        denominator += positive_similarities

        support += (
            positive_similarities > 0
        ).astype(int)

    cf_scores = (
        numerator /
        (denominator + denominator_smoothing)
    )

    # Require at least two positive similarity signals
    cf_scores[support < min_support] = np.nan

    # Exclude movies already rated by the user
    cf_scores.loc[
        cf_scores.index.intersection(
            list(user_ratings.keys())
        )
    ] = np.nan

    return cf_scores
