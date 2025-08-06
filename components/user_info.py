import streamlit as st
import pandas as pd
import os
import ast

def display_user_info(user_id, user_info):
    # Unpack values from tuple
    username, gender, age, genres, profile = user_info

    # Parse genres if stored as a JSON/text string
    if isinstance(genres, str):
        try:
            genres = ast.literal_eval(genres)  # Safely parse string to list
        except Exception:
            genres = [genres]  # Fallback to single genre as list
    if not isinstance(genres, list):
        genres = []

    # Display section
    with st.sidebar.expander("👤 我的資訊", expanded=True):
        st.markdown("### 📄 使用者資訊")
        st.write(f"🆔 使用者 ID: {user_id}")
        st.write(f"👤 使用者名稱: {username}")
        st.write(f"性別: {gender or '-'}")
        st.write(f"年齡: {age or '-'}")
        st.write(f"偏好類型: {', '.join(genres) if genres else '—'}")

        # Show user profile summary (generated from questionnaire)
        if profile:
            st.markdown("#### 🧠 偏好摘要")
            st.markdown(profile.strip())
