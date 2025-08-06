import streamlit as st
import os
from components.register import register_page
from components.login import login_page
from components.recommendations import show_recommendations
from components.browse import book_browsing
from components.sidebar import display_sidebar

# === APP ROUTING ===
st.set_page_config(page_title="Library Recommendation System", layout="wide")
st.title("📚 Library Recommendation System")

if "user" not in st.session_state:
    st.session_state.user = None

display_sidebar(st.session_state.user)

if st.session_state.user:
    # print("============ BOOK BROWSING RERUN ============")
    book_browsing()
else:
    page = st.sidebar.radio("請選擇操作", ["🔐 登入", "📋 註冊"])
    if page == "📋 註冊":
        register_page()
    else:
        login_page()