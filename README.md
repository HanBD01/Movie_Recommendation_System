# Movie Recommendation System

A personalized movie recommendation application built with **Python**, **MovieLens**, and **Streamlit**. The system combines content-based genre matching, item-based collaborative filtering, and popularity-aware movie scoring. Users can choose favorite genres, rate movies, and receive personalized Top-10 recommendations.

## Project Overview

This project covers the development of a recommendation system from initial dataset analysis to an interactive web application:

1. Explore MovieLens metadata and rating patterns.
2. Prepare movie features and weighted ratings.
3. Implement Content-Based Filtering using genre preferences.
4. Integrate explicit user ratings into personalized genre profiles.
5. Implement Item-Based Collaborative Filtering (CF) using Pearson correlation.
6. Combine signals into a popularity-aware hybrid recommender.
7. Evaluate models using chronological splits and ranking metrics.
8. Deploy the recommender through an English-language Streamlit interface.

### Dataset

| Statistic | Value |
|---|---:|
| Movies (`movies.csv`) | 9,742 |
| Rating records (`ratings.csv`) | 100,836 |
| Users | 610 |
| Genres used for modeling | 18 |

The project uses the MovieLens small dataset. The application reads the locally stored `data/movies.csv` and `data/ratings.csv` files.

## Project Structure

```text
movie_recommendation_v1/
├── data/
│   ├── movies.csv
│   └── ratings.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_content_based.ipynb
│   ├── 05_user_feedback.ipynb
│   ├── 06_collaborative_filtering.ipynb
│   ├── 07_hybrid_recommendation.ipynb
│   ├── 08_model_evaluation.ipynb
│   └── 09_streamlit_preparation.ipynb
├── recommender/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── content.py
│   ├── collaborative.py
│   └── hybrid.py
└── app_v2.py
```

- **`data/`** stores the original movie and rating data.
- **`notebooks/`** documents data exploration, algorithm development, experiments, and evaluation.
- **`recommender/`** contains reusable Python modules for the application.
- **`app_v2.py`** is the current English-language Streamlit application.

## Notebook Development Process

| Notebook | Purpose | Main Work |
|---|---|---|
| `01_data_understanding.ipynb` | Data understanding | Load data; inspect columns, movie/user identifiers, genres, and missing values. |
| `02_eda.ipynb` | Exploratory data analysis | Analyze rating distributions, ratings per user/movie, genre distributions, and data sparsity. |
| `03_preprocessing.ipynb` | Data preprocessing | One-hot encode genres, calculate rating statistics, apply weighted rating and Min-Max normalization. |
| `04_content_based.ipynb` | Content-Based Filtering | Construct genre profiles, calculate cosine similarity, and rank candidate movies. |
| `05_user_feedback.ipynb` | User feedback | Update genre preferences using the user's movie ratings relative to a fixed 3.5 baseline. |
| `06_collaborative_filtering.ipynb` | Item-Based Collaborative Filtering | Build user-item matrices, calculate Pearson similarities with shrinkage, and produce CF scores. |
| `07_hybrid_recommendation.ipynb` | Hybrid recommendation | Combine content and CF scores, handle unavailable CF signals, and generate Top-10 recommendations. |
| `08_model_evaluation.ipynb` | Evaluation and optimization | Compare models, test user-profile strategies, tune popularity weighting, and evaluate fallback coverage. |
| `09_streamlit_preparation.ipynb` | Application preparation | Verify that data-loading and recommendation modules work outside the exploratory notebooks. |

### 01 — Data Understanding

The initial notebook investigates the dataset structure and verifies the relationships among `userId`, `movieId`, movie titles, genres, and ratings. This establishes the foundation for later preprocessing.

### 02 — Exploratory Data Analysis

EDA examines user rating activity, movie rating counts, genre frequencies, and the uneven distribution of rating information. These findings motivate more robust movie quality scores and similarity adjustments.

### 03 — Data Preprocessing

Movie genres are converted into an 18-dimensional one-hot representation. A weighted rating incorporates both a movie's average rating and the amount of available rating evidence:

\[
WR = \frac{v}{v+m}R + \frac{m}{v+m}C
\]

Where:

- \(v\): number of ratings for the movie;
- \(R\): mean rating of the movie;
- \(C\): global mean rating;
- \(m=10\): rating-count adjustment parameter.

Weighted ratings are normalized using Min-Max scaling.

### 04 — Content-Based Recommendation

The content-based model computes cosine similarity \(G\) between the user genre profile and the movie genre vector, then combines it with the movie's normalized weighted rating \(P\):

\[
S_{\text{content}} = 0.7G + 0.3P
\]

Movies already rated by the user are excluded.

### 05 — User Feedback Integration

The user's explicit ratings contribute to the genre profile. A fixed rating baseline of 3.5 is used:

\[
\text{Preference} = \text{Rating} - 3.5
\]

Ratings above the baseline create positive preference signals; ratings below the baseline create negative signals. Each rating's contribution is distributed across the rated movie's genres.

### 06 — Collaborative Filtering

Item-Based CF uses historical movie rating co-occurrence and Pearson correlation calculated over users who rated both items.

| Setting | Value |
|---|---|
| Similarity | Pearson correlation |
| Minimum common raters | 10 |
| Shrinkage | \(n/(n+20)\) |
| Minimum positive similarity support | 2 |
| Preference rating baseline | 3.5 |
| Score denominator smoothing | 0.5 |

Similarity caching avoids repeated item-to-item computations.

### 07 — Hybrid Recommendation

The original hybrid model combines content scoring with a CF adjustment:

\[
S_{\text{hybrid}} =
\operatorname{clip}\left(S_{\text{content}} + 0.2\frac{CF}{1.5},\,0,\,1\right)
\]

Missing CF scores contribute a neutral adjustment of zero.

### 08 — Model Evaluation and Optimization

The notebook compares **Popularity**, **Content-Based**, **Collaborative Filtering**, and **Hybrid** approaches using:

- **Precision@10:** relevant recommendations divided by 10;
- **Recall@10:** fraction of held-out relevant movies retrieved;
- **NDCG@10:** ranking quality, emphasizing relevant items near the top;
- **Hit Rate@10:** fraction of users with at least one relevant recommendation;
- **Full-List Coverage@10:** fraction of users receiving all 10 recommendations.

Evaluation uses a per-user chronological training/test split. Further experiments use an inner chronological training/validation split. Relevant movies are those rated **4.0 or higher** in the held-out portion.

The project also evaluates:

- A fixed **3.5 rating baseline** against **User Mean Centering**;
- Popularity weighting \(\gamma\) from 0.0 to 1.0;
- CF contribution to ranking quality;
- Popularity Fallback for users with insufficient personalized candidates.

On the inner Validation set, \(\gamma=0.8\) yielded the highest **NDCG@10** among the tested popularity weights.

#### Original Test Results

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Popularity | **0.04443** | **0.03236** | **0.05550** |
| Content-Based | 0.01943 | 0.01477 | 0.02436 |
| Collaborative Filtering | 0.01385 | 0.01059 | 0.01479 |
| Hybrid | 0.02264 | 0.01662 | 0.02846 |
| Hybrid + Fallback | 0.02432 | 0.01791 | 0.03068 |

#### Improved Model Validation Results

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Content-Based | 0.01754 | 0.02176 | 0.02662 |
| Popularity-Aware Content | 0.03693 | 0.04111 | 0.05061 |
| Popularity-Aware Hybrid | 0.03929 | 0.04310 | 0.05108 |
| **Improved Hybrid + Fallback** | **0.04098** | **0.04396** | **0.05260** |

For **Improved Hybrid + Fallback**, the full Validation metrics were:

| Metric | Value |
|---|---:|
| Precision@10 | 0.040978 |
| Recall@10 | 0.043964 |
| NDCG@10 | 0.052600 |
| Hit Rate@10 | 0.259696 |
| Full-List Coverage@10 | 100% |
| Evaluated users | 593 |

**Interpretation:** The popularity-aware strategy improved Validation ranking metrics. Adding CF provided a small average NDCG@10 improvement over Popularity-Aware Content (approximately **0.000474**). Fallback provided a complete recommendation list for all evaluated users.

**Evaluation caveat:** Test results were examined before some later model-development decisions. Therefore, improved-model Validation results should be described as **exploratory**, not as an entirely independent final estimate of generalization performance. In addition, the application's new-user profile includes directly selected genres, which differs from the historical-user profiles used during offline evaluation.

### 09 — Streamlit Preparation

This notebook tests whether the independent `recommender` modules can load data, generate genre profiles, compute item similarities and CF scores, generate hybrid recommendations, and handle cold-start cases.

## The `recommender` Package

The `recommender` directory separates the recommendation engine from the UI. Its modules can be imported into notebooks or the Streamlit application.

### `data_loader.py` — Data Loading and Feature Engineering

**Functions:** `load_movie_data()`, `build_movie_features()`

- Reads `movies.csv` and `ratings.csv`.
- Calculates per-movie rating counts and means.
- Produces weighted and normalized movie ratings.
- Builds one-hot genre features.

### `content.py` — Content-Based Recommendation

**Functions:** `build_user_profile()`, `recommend_content_movies()`

- Builds user profiles from chosen genres and explicit movie ratings.
- Calculates genre cosine similarity.
- Combines content and popularity signals.
- Uses popularity fallback if too few movies match the user profile.

### `collaborative.py` — Collaborative Filtering

**Functions:** `build_user_item_matrix()`, `calculate_item_similarities()`, `calculate_cf_scores()`

- Builds a user-item rating matrix.
- Computes common-rater Pearson similarities with shrinkage.
- Aggregates rating-adjusted CF scores.
- Supports similarity caching.

### `hybrid.py` — Hybrid Recommendation

**Function:** `recommend_hybrid_movies()`

- Computes genre and popularity-aware scores.
- Incorporates CF scores where available.
- Excludes movies previously rated by the user.
- Fills incomplete recommendation lists using popularity-ranked movies.

The deployed scoring configuration is:

\[
S_{\text{improved}} =
\operatorname{clip}\left(0.14G + 0.86P + 0.2\frac{CF}{1.5},\,0,\,1\right)
\]

Where \(G\) is genre similarity and \(P\) is normalized weighted rating. The score combines popularity-aware content with a CF adjustment. The fallback ranks remaining candidates by weighted rating when fewer than 10 valid personalized candidates are available.

### `__init__.py` — Package Initialization

Organizes the recommendation functions as a Python package, allowing imports such as:

```python
from recommender.hybrid import recommend_hybrid_movies
```

## The Streamlit Application (`app_v2.py`)

`app_v2.py` provides the interactive **English-language** user interface. It calls the modules in `recommender/` to calculate and display recommendations.

### Main Features

1. **Select Your Favorite Genres** — Choose from 18 movie genres.
2. **Rate Movies** — Search by title and assign ratings from 0.5 to 5.0 in 0.5-point steps.
3. **My Rated Movies** — View or delete ratings recorded in the current session.
4. **Personalized Movie Recommendations** — Generate a Top-10 recommendation list.
5. **Recommendation Cards** — Display movie titles, genres, scores, recommendation sources, and explanations via **Why This Movie?**
6. **Session State** — Store ratings, recommendation results, and CF similarity cache during the session.

The current application does not include persistent user accounts or permanent storage of individual users' submitted ratings. Movie poster integration has not been completed.

## System Architecture

```text
MovieLens Dataset (movies.csv, ratings.csv)
                       |
                       v
                 data_loader.py
                       |
                       v
                 Movie Features
                       |
           +-----------+-----------+
           |                       |
           v                       v
       content.py          collaborative.py
           |                       |
           v                       v
    Genre Similarity         Item-Based CF
    + Popularity Score           Score
           |                       |
           +-----------+-----------+
                       |
                       v
                   hybrid.py
                       |
                       v
            Hybrid Recommendation
                       |
                       v
              Popularity Fallback
                       |
                       v
                    app_v2.py
                       |
                       v
             Personalized Top-10
             Movie Recommendations
```

The **notebooks** document the development process, **`recommender/`** provides the algorithm implementation, and **`app_v2.py`** connects that implementation to user interactions.

## Installation and Running the Application

### Prerequisites

- Conda
- Python 3.11
- The MovieLens CSV files under `data/`

### 1. Create the Conda Environment

```bash
conda create -n movie_recommendation python=3.11 -y
conda activate movie_recommendation
```

### 2. Install Dependencies

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipykernel streamlit plotly
```

### 3. Register the Notebook Kernel (Optional)

```bash
python -m ipykernel install --user --name movie_recommendation --display-name "Python (movie_recommendation)"
```

### 4. Run the English-Language Application

Open a terminal in the root project directory:

```bash
python -m streamlit run app_v2.py
```

Streamlit opens a local browser page for the application.

## Limitations and Future Improvements

- **Evaluation design:** The improved model was selected during exploratory Validation experiments after earlier Test results had been inspected. A new untouched holdout set or external evaluation would be needed for an independent final estimate.
- **CF benefit:** Collaborative Filtering made only a small average improvement over the popularity-aware content model during Validation.
- **Popularity influence:** The final ranking places substantial weight on normalized weighted ratings, which may favor widely rated, highly regarded movies.
- **Cold start:** The system uses popularity fallback when personalized recommendation signals are insufficient.
- **User persistence:** Rating inputs are maintained in Streamlit session state rather than stored permanently for user accounts.
- **Movie posters:** Poster retrieval and display have not yet been completed.
- **Potential enhancements:** More robust cold-start profiling, persistent rating storage, improved movie metadata matching, recommendation explanations, and broader out-of-sample evaluation.

## Conclusion

This project implements an end-to-end personalized movie recommendation workflow: MovieLens data exploration and preprocessing, content-based and collaborative recommendation development, hybrid scoring, evaluation, optimization, and an interactive Streamlit application.

The final application uses a **Popularity-Aware Hybrid Recommendation** strategy with a **Popularity Fallback** to generate Top-10 movie recommendations based on user genre preferences and movie ratings. Its modular architecture keeps experimentation (`notebooks/`), reusable algorithms (`recommender/`), and user interaction (`app_v2.py`) distinct and maintainable.
