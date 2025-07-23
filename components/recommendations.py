import streamlit as st
import pandas as pd
from components.utils import load_books, fetch_image_cached

DEFAULT_IMAGE_PATH = "data/default_cover.jpeg"

def get_top_n_recommendations(user_id, n=5):
    # Load the full book database
    full_books = load_books()
    books = full_books.sample(frac=1).reset_index(drop=True)
    recommended_books = books.iloc[0:n]

    return recommended_books

def show_recommendations(user_id, recommended_books_df):
    st.subheader("🎯 推薦書單")

    if recommended_books_df.empty:
        st.info("目前沒有推薦結果。")
        return

    st.markdown("根據您的偏好推薦的書籍：")

    # Use horizontal scrollable container
    with st.container():
        cols = st.columns(len(recommended_books_df))
        for i, book in recommended_books_df.iterrows():
            with cols[i]:
                image_url = book.get('bi_image')
                if pd.notna(image_url):
                    st.image(fetch_image_cached(image_url), use_container_width=True)
                else:
                    st.image(DEFAULT_IMAGE_PATH, use_container_width=True)
                st.caption(book["bi_title"])
                if st.button("查看", key=f"rec_view_{i}"):
                    st.session_state["viewed_book_id"] = book["id"]
