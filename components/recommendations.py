import streamlit as st
import pandas as pd
from components.utils import load_books, fetch_image_cached
from components.interactions import log_interaction

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

    # Button to refresh recommendations
    if st.button("🔄 更新推薦結果"):
        st.session_state["force_refresh_recs"] = True
        
    st.markdown("根據您的偏好推薦的書籍：")

    # Use horizontal scrollable container
    with st.container():
        cols = st.columns(len(recommended_books_df))
        for idx, book in recommended_books_df.iterrows():
            with cols[idx]:
                image_url = book.get('bi_image')
                if pd.notna(image_url):
                    st.image(fetch_image_cached(image_url), use_container_width=True)
                else:
                    st.image(DEFAULT_IMAGE_PATH, use_container_width=True)
                st.caption(book["bi_title"])
                if st.button("瀏覽", key=f"rec_{book['bi_id']}_{idx}"):
                    log_interaction(st.session_state.user, book, "瀏覽")
                    st.session_state.expanded_books.add(book["bi_id"])
                    st.rerun()  # 避免按鈕殘留影響下一輪行為

                    # 自動展開
                    if book["bi_id"] in st.session_state.expanded_books:
                        with st.expander("更多資訊", expanded=True):
                            st.write(f"📌 書名：{book['bi_title']}")
                            st.write(f"📌 編著者：{book['bi_auther']}")
                            st.write(f"📌 索書號：{book.get('bi_class', '—')}")
                            st.write(f"📌 索書號分類：{book.get('category', '—')}")
                            st.write(f"📌 ISBN：{book.get('bi_isbn', '—')}")
                            st.write(f"📌 出版資訊：{book.get('bi_publisher', '—')}")
                            st.write(f"📌 出版年：{book.get('bi_publish_year', '—')}")
                            st.write(f"📌 館藏地：{book.get('bi_site', '—')}")
                            st.write(f"📌 大綱: {book.get('bi_content', '-')}")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("收藏", key=f"fav_{book['bi_id']}_{idx}"):
                                    log_interaction(st.session_state.user, book, "收藏")
                                    st.rerun()
                            with col2:
                                if st.button("預約", key=f"reserve_{book['bi_id']}_{idx}"):
                                    log_interaction(st.session_state.user, book, "預約")
                                    st.rerun()
                            with col3:
                                if st.button("借閱", key=f"borrow_{book['bi_id']}_{idx}"):
                                    log_interaction(st.session_state.user, book, "借閱")
                                    st.rerun()