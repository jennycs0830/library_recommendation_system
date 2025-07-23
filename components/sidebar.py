import streamlit as st
import json
import os

from components.user_info import display_user_info
from components.interactions import display_interaction_history
from components.utils import load_users

DATA_FOLDER = "data"
INTERACTIONS_DB = os.path.join(DATA_FOLDER, "interactions.csv")

def display_sidebar(user_id):
    st.sidebar.title("👤 我的資訊 / 行為紀錄")

    if user_id:
        users = load_users()
        user_info = users.get(user_id, {})
        if user_info:
            display_user_info(user_id, user_info)
            display_interaction_history(user_id, INTERACTIONS_DB)
        else:
            st.sidebar.info("使用者資訊不存在")

        if st.sidebar.button("🚪 登出"):
            st.session_state.user = None
            st.rerun()
    else:
        st.sidebar.info("尚未登入")
