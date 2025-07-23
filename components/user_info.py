import streamlit as st
import pandas as pd
import os

def display_user_info(user_id, user_info):
    st.sidebar.expander("👤 我的資訊", expanded=True)
    st.sidebar.markdown("### 📄 使用者資訊")
    st.sidebar.write(f"🆔 使用者 ID: {user_id}")
    st.sidebar.write(f"性別: {user_info.get('gender', '-')}")
    st.sidebar.write(f"年齡: {user_info.get('age', '-')}")
    st.sidebar.write(f"偏好類型: {user_info.get('preferred_genres', [])}")

    if 'questionnaire' in user_info:
        st.sidebar.markdown("#### 📋 偏好問卷摘要")
        for k, v in user_info['questionnaire'].items():
            if v.strip():
                st.sidebar.markdown(f"• **{v.strip()}**")