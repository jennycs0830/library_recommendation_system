import streamlit as st
import json
import os
import psycopg2
import openai
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from components.utils import encode_user_embedding, get_pg_connection

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

@ st.cache_resource
def load_embedding_model():
    model = SentenceTransformer('shibing624/text2vec-base-chinese')
    return model

embedding_model = load_embedding_model()

def user_exists(username):
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE username = %s", (username, ))
            return cur.fetchone() is not None

def insert_user(username, user_profile, user_embedding, gender, age, genres):
    create_user = """
        INSERT INTO users (username, user_gender, user_age, user_genres, user_profile, user_embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING user_id;
    """
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            genres = ", ".join(genres)
            values = (username, gender, age, genres, user_profile, user_embedding)
            cur.execute(create_user, values)
            user_id = cur.fetchone()[0]
            conn.commit()
            print(f"User {username} created successfully with user_id {user_id}.")

    return user_id

def generate_user_profile(gender, age, genres, q1, q2, q3, q4):
    prompt_template = f"""
    你是一個圖書推薦系統的助手，請根據以下使用者基本資料與問卷回答，建立一段語意化的閱讀偏好檔案，並列出使用者可能感興趣的主題與風格關鍵字。

    ---

    使用者基本資料：
    - 性別: {gender}
    - 年齡層: {age}
    - 偏好書籍類型: {", ".join(genres)}

    問卷回答：
    1. 喜歡的書籍與原因：{q1}
    2. 閱讀偏好的風格或主題：{q2}
    3. 想閱讀的主題或內容：{q3}
    4. 閱讀的價值與意義：{q4}

    ---

    請執行以下兩項任務：

    ### 一、請用自然且具語意的語氣，撰寫一段使用者的閱讀風格與偏好描述，不需重複列出題目與答案。請綜合所有資訊，具體說明此人偏好什麼樣的書、閱讀動機為何、適合什麼主題與風格等。請避免逐條回答，而是寫成一段具有邏輯與語感的文字。

    ### 二、請根據你的分析，列出 5～10 個與此使用者高度相關的「閱讀偏好關鍵字」，這些可以是主題（如：心理學）、風格（如：黑色幽默）、價值觀（如：自我成長）、書籍形式（如：報導文學）等。請以 bullet point 條列方式呈現，內容需具語意與概括性，避免只是重複原句中的片段。

    """

    response = openai.ChatCompletion.create(
        model = "gpt-4.1-mini",
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates user profiles based on their reading preferences."},
            {"role": "user", "content": prompt_template}
        ],
        temperature = 0.7,
        max_tokens = 768
    )
    user_profile = response.choices[0].message.content.strip()
    print(f"Generated user profile: {user_profile}")

    # user_profile = "AI 人工智慧 軟體工程師 演算法開發 AI模型訓練 AI 人工智慧 軟體工程師 演算法開發 AI模型訓練 AI 人工智慧 軟體工程師 演算法開發 AI模型訓練 AI 人工智慧 軟體工程師 演算法開發 AI模型訓練 "
    user_embedding = embedding_model.encode(user_profile, convert_to_tensor=True).numpy().tolist()
    encoded_embedding = encode_user_embedding(user_embedding)

    return user_profile, encoded_embedding

def register_page():
    st.subheader("📋 使用者註冊")
    with st.form("register_form"):
        username = st.text_input("使用者帳號", placeholder="username")
        gender = st.selectbox("性別", ["男", "女"], placeholder="gender")
        age = st.selectbox("年齡", ["兒童", "少年", "青年", "壯年", "老年"], placeholder="age")
        genres = st.multiselect("偏好類型", ["文學", "社會科學", "自然科學", "兒童", "歷史", "哲學", "藝術", "科技", "語言"], placeholder="genres")

        st.markdown("---")
        st.markdown("### 📋 偏好問卷 (選填)")
        q1 = st.text_area("📖 請描述一本你印象深刻或喜歡的書籍，以及原因")
        q2 = st.text_area("🧠 你閱讀時偏好哪種風格或主題？（如：輕鬆幽默、深度思考、實用工具書等）")
        q3 = st.text_area("💡 若你可以選擇下一本想讀的書，會是關於什麼主題或內容？")
        q4 = st.text_area("🌱 閱讀對你而言最有價值的是什麼？（例如：獲得知識、情感陪伴、開拓視野等）")

        submit = st.form_submit_button("註冊")
        if submit:
            print(f"Attempting to register user: {username}")

            if not user_exists(username):
                user_profile, user_embeds = generate_user_profile(gender, age, genres, q1, q2, q3, q4)
                user_id = insert_user(username, user_profile, user_embeds, gender, age, genres)
                st.success("註冊成功！請登入使用。")
                print(f"User {username} registered successfully.")
                st.session_state.user = user_id
                print(f"Session state updated with user: {st.session_state.user}")
                st.rerun()
            else:
                st.error("此使用者 ID 已被註冊！")
