import streamlit as st
import pandas as pd
import psycopg2
import numpy as np
import faiss

from components.utils import fetch_image_cached, get_pg_connection, get_book_metadata
from components.interactions import log_interaction

DEFAULT_IMAGE_PATH = "data/default_cover.jpeg"
FAISS_INDEX_PATH = "database/faiss_books.index"
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

def get_user_embedding(user_id):
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_embedding FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row:
                return np.array(row[0], dtype=np.float32)
            else:
                return None

def get_top_n_recommendations(user_id, n=10):
    user_embedding = get_user_embedding(user_id)
    print(f"user_embedding, shape: {user_embedding.shape}")
    print(f"faiss index, dim {faiss_index.d}")
    D, I = faiss_index.search(user_embedding.reshape(1, -1), n)
    return I[0].tolist()

def show_recommendations(user_id):
    st.subheader("🎯 推薦書單")

    if "expanded_books" not in st.session_state:
        st.session_state.expanded_books = set()

    # First check if this is the first load or a forced refresh
    if "force_refresh_recs" not in st.session_state:
        st.session_state["force_refresh_recs"] = True  # initial load

    # Button to refresh recommendations
    if st.button("🔄 更新推薦結果"):
        st.session_state["force_refresh_recs"] = True

    # Only call recommendation logic when needed
    if st.session_state["force_refresh_recs"]:
        recommended_ids = get_top_n_recommendations(user_id, 5)
        st.session_state["recommended_ids"] = recommended_ids
        st.session_state["force_refresh_recs"] = False  # reset flag
    else:
        recommended_ids = st.session_state.get("recommended_ids", [])

    if not recommended_ids:
        st.info("目前沒有推薦結果。")
        return

    st.markdown("根據您的偏好推薦的書籍：")
    books_metadata = get_book_metadata(recommended_ids)

    # Use horizontal scrollable container
    with st.container():
        cols = st.columns(len(recommended_ids))
        for idx, book_info in enumerate(books_metadata):
            with cols[idx]:
                image_url = book_info['image_url']
                if pd.notna(image_url):
                    st.image(fetch_image_cached(image_url), use_container_width=True)
                else:
                    st.image(DEFAULT_IMAGE_PATH, use_container_width=True)
                st.caption(book_info["title"])
                if st.button("瀏覽", key=f"rec_{book_info['book_id']}_{idx}"):
                    log_interaction(st.session_state.user, book_info["book_id"], "瀏覽")
                    st.session_state.expanded_books.add(book_info["book_id"])
                    # st.rerun()

                # 自動展開
                if book_info["book_id"] in st.session_state.expanded_books:
                    with st.expander("更多資訊", expanded=True):
                        st.write(f"📌 書名：{book_info['title']}")
                        st.write(f"📌 編著者：{book_info['author']}")
                        st.write(f"📌 索書號：{book_info.get('call_number', '—')}")
                        st.write(f"📌 索書號分類：{book_info.get('category', '—')}")
                        st.write(f"📌 ISBN：{book_info.get('isbn', '—')}")
                        st.write(f"📌 出版資訊：{book_info.get('publisher', '—')}")
                        st.write(f"📌 出版年：{book_info.get('publisher_year', '—')}")
                        st.write(f"📌 館藏地：{book_info.get('site', '—')}")
                        st.write(f"📌 大綱: {book_info.get('content', '-')}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("收藏", key=f"fav_{book_info['book_id']}_{idx}"):
                                log_interaction(st.session_state.user, book_info["book_id"], "收藏")
                                st.rerun()
                        with col2:
                            if st.button("預約", key=f"reserve_{book_info['book_id']}_{idx}"):
                                log_interaction(st.session_state.user, book_info["book_id"], "預約")
                                st.rerun()
                        with col3:
                            if st.button("借閱", key=f"borrow_{book_info['book_id']}_{idx}"):
                                log_interaction(st.session_state.user, book_info["book_id"], "借閱")
                                st.rerun()
