import streamlit as st
import json
import os
from components.utils import load_users, save_users

def register_page():
    st.subheader("📋 使用者註冊")
    with st.form("register_form"):
        user_id = st.text_input("使用者 ID")
        gender = st.selectbox("性別", ["男", "女"])
        age = st.selectbox("年齡", ["兒童", "少年", "青年", "壯年", "老年"])
        genres = st.multiselect("偏好類型", ["文學", "社會科學", "自然科學", "兒童", "歷史", "哲學", "藝術", "科技", "語言"])

        st.markdown("---")
        st.markdown("### 📋 偏好問卷 (選填)")
        q1 = st.text_area("📖 請描述一本你印象深刻或喜歡的書籍，以及原因")
        q2 = st.text_area("🧠 你閱讀時偏好哪種風格或主題？（如：輕鬆幽默、深度思考、實用工具書等）")
        q3 = st.text_area("💡 若你可以選擇下一本想讀的書，會是關於什麼主題或內容？")
        q4 = st.text_area("🌱 閱讀對你而言最有價值的是什麼？（例如：獲得知識、情感陪伴、開拓視野等）")

        submit = st.form_submit_button("註冊")
        if submit:
            print(f"Attempting to register user: {user_id}")
            users = load_users()
            if user_id in users:
                st.error("此使用者 ID 已被註冊！")
            else:
                users[user_id] = {
                    "gender": gender,
                    "age": age,
                    "preferred_genres": genres,
                    "questionnaire": {
                        "q1": q1,
                        "q2": q2,
                        "q3": q3,
                        "q4": q4
                    }
                }
                save_users(users)
                st.success("註冊成功！請登入使用。")
                print(f"User {user_id} registered successfully.")
                st.session_state.user = user_id
                print(f"Session state updated with user: {st.session_state.user}")
                st.rerun()