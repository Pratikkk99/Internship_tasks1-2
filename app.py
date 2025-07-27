import streamlit as st
from recommender import load_data, recommend_by_genre

st.title("🎬 Genre-Based Movie Recommender")
st.markdown("Get movie suggestions based on your favorite genre!")

# Load data
movies_df = load_data()

# Genre selection
genre = st.selectbox("Choose a genre:", 
    ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
     'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
     'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

if st.button("Recommend"):
    results = recommend_by_genre(movies_df, genre)
    if not results.empty:
        st.subheader(f"Top {len(results)} {genre} movies:")
        for i, row in results.iterrows():
            st.write(f"- {row['title']} ({row['genres']})")
    else:
        st.warning("No movies found for this genre.")
