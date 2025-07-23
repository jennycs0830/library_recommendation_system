import streamlit as st
import json
import os
from components.utils import load_users

def login_page():
    st.subheader("🔐 使用者登入")
    with st.form("login_form"):
        user_id = st.text_input("使用者 ID")
        submit = st.form_submit_button("登入")
        if submit:
            users = load_users()
            if user_id in users:
                st.session_state.user = user_id
                st.success("登入成功！")
                st.rerun()
            else:
                st.error("無此使用者，請先註冊")