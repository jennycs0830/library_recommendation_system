from components.utils import fetch_image_cached, get_pg_connection, DEFAULT_IMAGE_PATH
from components.recommendations import show_recommendations
from components.interactions import log_interaction
import streamlit as st
import pandas as pd

def fetch_all_books():
    query = """
    SELECT 
        book_id, title, author, call_number, isbn, publisher,
        publisher_year, site, category, content, image_url
    FROM books;
    """
    with get_pg_connection() as conn:
        return pd.read_sql_query(query, conn)

def book_browsing():
    user_id = st.session_state.get("user", None)
    show_recommendations(user_id)

    if "shuffled_books" not in st.session_state:
        st.session_state.shuffled_books = fetch_all_books().sample(frac=1).reset_index(drop=True)

    books = st.session_state.shuffled_books
    page_size = 20
    total_pages = (len(books) - 1) // page_size + 1
    page_num = st.session_state.get("page_num", 0)
    current_books = books.iloc[page_num * page_size: (page_num + 1) * page_size]

    if "expanded_books" not in st.session_state:
        st.session_state.expanded_books = set()

    st.subheader("🎯 全部書籍")
    for idx, book in current_books.iterrows():
        with st.container():
            cols = st.columns([1, 4])
            with cols[0]:
                image_url = book.get("image_url")
                if pd.notna(image_url):
                    st.image(fetch_image_cached(image_url), use_container_width=True)
                else:
                    st.image(DEFAULT_IMAGE_PATH, use_container_width=True)

            with cols[1]:
                st.markdown(f"**{book['title']}**")
                if st.button("瀏覽", key=f"view_{book['book_id']}_{idx}"):
                    log_interaction(user_id, book["book_id"], "瀏覽")
                    st.session_state.expanded_books.add(book["book_id"])
                    st.rerun()

                if book["book_id"] in st.session_state.expanded_books:
                    with st.expander("更多資訊", expanded=True):
                        st.write(f"📌 書名：{book['title']}")
                        st.write(f"📌 編著者：{book['author']}")
                        st.write(f"📌 索書號：{book.get('call_number', '—')}")
                        st.write(f"📌 索書號分類：{book.get('category', '—')}")
                        st.write(f"📌 ISBN：{book.get('isbn', '—')}")
                        st.write(f"📌 出版資訊：{book.get('publisher', '—')}")
                        st.write(f"📌 出版年：{book.get('publisher_year', '—')}")
                        st.write(f"📌 館藏地：{book.get('site', '—')}")
                        st.write(f"📌 大綱: {book.get('content', '-')}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("收藏", key=f"fav_{book['book_id']}_{idx}"):
                                log_interaction(user_id, book["book_id"], "收藏")
                                st.rerun()
                        with col2:
                            if st.button("預約", key=f"reserve_{book['book_id']}_{idx}"):
                                log_interaction(user_id, book["book_id"], "預約")
                                st.rerun()
                        with col3:
                            if st.button("借閱", key=f"borrow_{book['book_id']}_{idx}"):
                                log_interaction(user_id, book["book_id"], "借閱")
                                st.rerun()

    # Pagination
    col1, col2, col3 = st.columns([1, 2, 1])
    if col1.button("⬅️ 上一頁") and page_num > 0:
        st.session_state.page_num = page_num - 1
        st.rerun()
    col2.markdown(f"<div style='text-align:center;'>第 {page_num + 1} 頁 / 共 {total_pages} 頁</div>", unsafe_allow_html=True)
    if col3.button("➡️ 下一頁") and page_num < total_pages - 1:
        st.session_state.page_num = page_num + 1
        st.rerun()
