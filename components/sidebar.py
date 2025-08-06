import streamlit as st
import json
import os

from components.user_info import display_user_info
from components.interactions import display_interaction_history
from components.utils import get_pg_connection

def display_sidebar(user_id):
    st.sidebar.title("📋 我的帳戶")

    if user_id:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT username, user_gender, user_age, user_genres, user_profile 
                    FROM users WHERE user_id = %s
                """, (user_id,))
                user_info = cur.fetchone()

                if user_info:
                    st.sidebar.markdown("### 👤 使用者資訊")
                    display_user_info(user_id, user_info)

                    st.sidebar.markdown("### 📚 行為紀錄")
                    display_interaction_history(user_id)
                else:
                    st.sidebar.info("使用者資訊不存在")

        if st.sidebar.button("🚪 登出"):
            st.session_state.user = None
            st.rerun()
    else:
        st.sidebar.info("尚未登入")
