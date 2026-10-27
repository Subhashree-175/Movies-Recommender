```python
import ast
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Discover movies similar to your favourites!")

@st.cache_data
def load_data():
    def read_csv_file(filename):
        path = Path(filename)

        if path.exists():
            return pd.read_csv(path)

        zip_path = Path(filename + ".zip")
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as archive:
                csv_name = next(
                    name for name in archive.namelist()
                    if name.endswith(".csv")
                )
                with archive.open(csv_name) as file:
                    return pd.read_csv(file)

        raise FileNotFoundError(f"Dataset not found: {filename}")

    movies = read_csv_file("tmdb_5000_movies.csv")
    credits = read_csv_file("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    def parse_names(value):
        try:
            return [item["name"] for item in ast.literal_eval(value)]
        except (ValueError, SyntaxError, TypeError, KeyError):
            return []

    movies["genres"] = movies["genres"].apply(parse_names)
    movies["keywords"] = movies["keywords"].apply(parse_names)
    movies["cast"] = movies["cast"].apply(
        lambda value: parse_names(value)[:3]
    )

    def get_directors(value):
        try:
            crew = ast.literal_eval(value)
            return [
                person["name"].replace(" ", "")
                for person in crew
                if person.get("job") == "Director"
            ]
        except (ValueError, SyntaxError, TypeError):
            return []

    movies["crew"] = movies["crew"].apply(get_directors)
    movies["overview"] = movies["overview"].fillna("").apply(
        lambda value: value.split()
    )

    for column in ["genres", "keywords", "cast"]:
        movies[column] = movies[column].apply(
            lambda items: [
                item.replace(" ", "") for item in items
            ]
        )

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    newdf = movies[["movie_id", "title", "tags"]].copy()
    newdf["tags"] = newdf["tags"].apply(lambda items: " ".join(items))

    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )
    vectors = vectorizer.fit_transform(newdf["tags"])

    return newdf, vectors

try:
    newdf, vectors = load_data()
except Exception as error:
    st.error("Unable to load the movie datasets.")
    st.exception(error)
    st.stop()

selected_movie = st.selectbox(
    "Choose a movie",
    sorted(newdf["title"].dropna().unique())
)

if st.button("🍿 Recommend Movies"):
    movie_index = newdf.index[
        newdf["title"] == selected_movie
    ][0]

    similarities = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    recommendations = similarities.argsort()[::-1]
    recommendations = [
        index for index in recommendations
        if index != movie_index
    ][:5]

    st.subheader("✨ Movies you may like")

    columns = st.columns(5)

    for column, index in zip(columns, recommendations):
        with column:
            st.markdown("### 🎬")
            st.write(newdf.iloc[index]["title"])
```
