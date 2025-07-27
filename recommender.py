import pandas as pd

# Load movies data
def load_data():
    movies = pd.read_csv("movies.csv")
    return movies

# Recommend movies by genre
def recommend_by_genre(movies_df, genre, top_n=5):
    genre_movies = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)]
    top_movies = genre_movies.sample(n=min(top_n, len(genre_movies)))  # Random top 5
    return top_movies[['title', 'genres']]
