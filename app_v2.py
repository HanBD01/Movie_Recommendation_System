
import streamlit as st
import pandas as pd

from recommender.data_loader import (
    load_movie_data,
    build_movie_features,
    AVAILABLE_GENRES
)

from recommender.content import build_user_profile
from recommender.collaborative import build_user_item_matrix
from recommender.hybrid import recommend_hybrid_movies


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write(
    "Discover personalized movie recommendations "
    "based on your favorite genres and movie ratings."
)


# --------------------------------------------------
# 2. Data loading
# --------------------------------------------------

@st.cache_data
def load_data():
    movies, ratings = load_movie_data()

    movie_features = build_movie_features(
        movies=movies,
        ratings=ratings,
        m=10
    )

    return movies, ratings, movie_features


@st.cache_resource
def load_cf_matrix():
    _, ratings = load_movie_data()
    return build_user_item_matrix(ratings)


movies, ratings, movie_features = load_data()
user_item_matrix = load_cf_matrix()


# --------------------------------------------------
# 3. User session state
# --------------------------------------------------

if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

if "similarity_cache" not in st.session_state:
    st.session_state.similarity_cache = {}

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None


# --------------------------------------------------
# 4. Favorite genre selection
# --------------------------------------------------

st.subheader("1. Select Your Favorite Genres")

def clear_recommendations():
    st.session_state.recommendations = None


selected_genres = st.multiselect(
    "Choose your favorite movie genres.",
    options=AVAILABLE_GENRES,
    default=["Action", "Sci-Fi"],
    on_change=clear_recommendations
)


# --------------------------------------------------
# 5. Movie search and rating
# --------------------------------------------------

st.subheader("2. Rate Movies")

search_keyword = st.text_input(
    "Search for a movie to rate.",
    placeholder="e.g., Matrix, Toy Story, Titanic"
)

if search_keyword.strip():

    matched_movies = movies[
        movies["title"].str.contains(
            search_keyword.strip(),
            case=False,
            na=False,
            regex=False
        )
    ].copy()

    if matched_movies.empty:
        st.info("No movies found.")

    else:
        selected_movie_id = st.selectbox(
            "Select a movie.",
            options=matched_movies["movieId"].tolist(),
            format_func=lambda movie_id: (
                matched_movies.loc[
                    matched_movies["movieId"] == movie_id,
                    "title"
                ].iloc[0]
            )
        )

        current_rating = st.session_state.user_ratings.get(
            selected_movie_id, 3.5
        )

        rating = st.select_slider(
            "Select your rating.",
            options=[
                0.5, 1.0, 1.5, 2.0, 2.5,
                3.0, 3.5, 4.0, 4.5, 5.0
            ],
            value=current_rating
        )

        if st.button("Save Rating"):
            st.session_state.user_ratings[
                selected_movie_id
            ] = rating

            st.session_state.recommendations = None

            st.success("Your rating has been saved.")


# --------------------------------------------------
# 6. Rated movie list
# --------------------------------------------------

st.subheader("3. My Rated Movies")

if not st.session_state.user_ratings:

    st.info("You haven't rated any movies yet.")

else:

    rated_df = pd.DataFrame([
        {
            "movieId": movie_id,
            "rating": rating
        }
        for movie_id, rating
        in st.session_state.user_ratings.items()
    ])

    rated_df = rated_df.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    st.dataframe(
        rated_df[
            ["title", "rating"]
        ].rename(columns={
            "title": "Movie",
            "rating": "My Rating"
        }),
        hide_index=True,
        use_container_width=True
    )

    remove_movie_id = st.selectbox(
        "Select a movie to remove",
        options=list(st.session_state.user_ratings.keys()),
        format_func=lambda movie_id: (
            movies.loc[
                movies["movieId"] == movie_id,
                "title"
            ].iloc[0]
        )
    )

    if st.button("Remove Rating"):
        del st.session_state.user_ratings[remove_movie_id]
        st.session_state.recommendations = None
        st.rerun()


# --------------------------------------------------
# 7. Hybrid movie recommendations
# --------------------------------------------------

st.subheader("4. Personalized Movie Recommendations")

if st.button(
    "🎯 Get Recommendations",
    type="primary",
    use_container_width=True
):

    user_profile = build_user_profile(
        selected_genres=selected_genres,
        user_ratings=st.session_state.user_ratings,
        movie_features=movie_features,
        available_genres=AVAILABLE_GENRES
    )

    with st.spinner("Generating your recommendations..."):

        recommendations = recommend_hybrid_movies(
            user_profile=user_profile,
            user_ratings=st.session_state.user_ratings,
            movie_features=movie_features,
            available_genres=AVAILABLE_GENRES,
            user_item_matrix=user_item_matrix,
            similarity_cache=st.session_state.similarity_cache,
            gamma=0.8,
            beta=0.2,
            k=10
        )

    st.session_state.recommendations = recommendations




# --------------------------------------------------
# 8. Display Recommendation Cards
# --------------------------------------------------

if st.session_state.recommendations is not None:

    recommendations = st.session_state.recommendations

    if recommendations.empty:
        st.warning("No recommendations are available.")

    else:
        st.success(
            f"Successfully generated {len(recommendations)} "
            "movie recommendations."
        )

        st.subheader("🎬 Top 10 Recommended Movies")

        col1, col2 = st.columns(2)

        for i, (_, movie) in enumerate(
            recommendations.iterrows()
        ):

            column = col1 if i % 2 == 0 else col2

            with column:

                with st.container(border=True):

                    st.markdown(
                        f"### {i + 1}. {movie['title']}"
                    )

                    genre_text = movie["genres"].replace(
                        "|", " · "
                    )

                    st.caption(genre_text)

                    score = float(
                        movie["recommendation_score"]
                    )

                    st.write(
                        f"**Recommendation Score: {score:.4f}**"
                    )

                    st.progress(
                        min(max(score, 0.0), 1.0)
                    )

                    source = movie[
                        "recommendation_source"
                    ]

                    if source == "Hybrid":
                        st.caption(
                            "Recommendation Method: Hybrid"
                        )
                    else:
                        st.caption(
                            "Recommendation Method: Popularity Fallback"
                        )

                    with st.expander("Why This Movie?"):

                        if source == "Popularity Fallback":

                            st.write(
                                "This movie was recommended based on "
                                "its normalized weighted rating "
                                "because there were not enough "
                                "personalized recommendation candidates."
                            )

                        else:

                            movie_genres = set(
                                movie["genres"].split("|")
                            )

                            matched_genres = [
                                genre
                                for genre in selected_genres
                                if genre in movie_genres
                            ]

                            if matched_genres:
                                st.write(
                                    "Matches your favorite genres: "
                                    + ", ".join(matched_genres)
                                )

                            else:
                                st.write(
                                    "Recommended based on your "
                                    "personalized genre profile."
                                )

                            st.write(
                                "Genre Similarity: "
                                f"{movie['genre_similarity']:.4f}"
                            )

                            if pd.notna(movie["cf_score"]):

                                st.write(
                                    "Collaborative Filtering Score: "
                                    f"{movie['cf_score']:.4f}"
                                )

                            else:

                                st.write(
                                    "No collaborative filtering score "
                                    "was available for this movie."
                                )

                        st.write(
                            "Normalized Weighted Rating: "
                            f"{movie['normalized_rating']:.4f}"
                        )
