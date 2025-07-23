import streamlit as st
import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO

from components.interactions import log_interaction
from components.recommendations import get_top_n_recommendations, show_recommendations
from components.utils import load_books, fetch_image_cached, DEFAULT_IMAGE_PATH

def book_browsing():
    user_id = st.session_state.get("user", None)
    recommended_books = get_top_n_recommendations(user_id, n=5)
    show_recommendations(user_id, recommended_books)

    full_books = load_books()
    books = full_books.sample(frac=1).reset_index(drop=True)

    page_size = 20
    total_pages = (len(books) - 1) // page_size + 1
    page_num = st.session_state.get("page_num", 0)
    current_books = books.iloc[page_num * page_size: (page_num + 1) * page_size]

    # Track which book is expanded
    if "expanded_books" not in st.session_state:
        st.session_state.expanded_books = set()

    st.subheader("🎯 全部書籍")
    for idx, book in current_books.iterrows():
        with st.container():
            cols = st.columns([1, 4])
            with cols[0]:
                image_url = book.get('bi_image')
                if pd.notna(image_url):
                    st.image(fetch_image_cached(image_url), use_container_width=True)
                else:
                    st.image(DEFAULT_IMAGE_PATH, use_container_width=True)

            with cols[1]:
                st.markdown(f"**{book['bi_title']}**")
                # 這邊是新的「瀏覽」按鈕行為
                if st.button("瀏覽", key=f"view_{book['bi_id']}_{idx}"):
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


    # Pagination buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    if col1.button("⬅️ 上一頁") and page_num > 0:
        st.session_state.page_num = page_num - 1
        st.rerun()
    col2.markdown(f"<div style='text-align:center;'>第 {page_num + 1} 頁 / 共 {total_pages} 頁</div>", unsafe_allow_html=True)
    if col3.button("➡️ 下一頁") and page_num < total_pages - 1:
        st.session_state.page_num = page_num + 1
        st.rerun()
