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
        INSERT INTO users (username, user_gender, user_age, user_genres, user_profile, user_embedding_cur, user_embedding_prev)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING user_id;
    """
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            genres = ", ".join(genres)
            values = (username, gender, age, genres, user_profile, user_embedding, [])
            cur.execute(create_user, values)
            user_id = cur.fetchone()[0]
            conn.commit()
            print(f"User {username} created successfully with user_id {user_id}.")

    return user_id

def generate_user_profile(gender, age, genres, q1, q2, q3, q4):
    prompt_template = f"""
    你是一位有才華的創作家，專門為特定讀者群撰寫書籍，擅長依據讀者的興趣與需求構思精采的書名與吸引人的大綱。

    ## 讀者資訊
    - 年齡: {age}
    - 性別: {gender}
    - 偏好類型: {genres}

    ## 使用者問卷回答
    1. 印象深刻或喜歡的書籍與原因: {q1}
    2. 閱讀時偏好風格或主題: {q2}
    3. 下一本想讀的書主題或內容: {q3}
    4. 閱讀對你最有價值的地方: {q4}

    ## 生成要求
    - 根據以上資訊，生成 **3 本書** 的推薦範例
    - 每本書須包含：
    1. **書名**（吸引人、符合讀者品味）
    2. **大綱**（不少於 50 字，避免過於簡略，應有情節、背景、主題）
    - 語氣與內容需符合 {genres} 類型
    - 適合 {age} 歲 {gender} 的讀者
    - 保持創意，但需貼近讀者可能喜愛的題材

    ## 參考範例
    1. 書名: 青石街的午後  
    大綱: 在舊城區的一條青石街上，開著一家小咖啡館的女子，每日觀察街上形形色色的路人。一次偶然的相遇，讓她卷入一位失語畫家的故事。畫布上的顏色，訴說著無人知曉的孤獨與渴望。細膩的筆觸描繪生活的縫隙，探討人與人之間微妙的連結與治癒。

    2. 書名: 第十三封信  
    大綱: 一名圖書館管理員在整理舊藏書時，發現一本書中夾著十二封匿名信，信中預告了一連串的失蹤事件。當第十三封信出現時，信上寫的名字竟是她自己。她必須在時間耗盡前，破解字裡行間的暗號，揭開隱藏在城市深處的真相。

    3. 書名: 流光編織者  
    大綱: 在未來的極地城市，一位年輕的編織師擁有將光線編成實體的能力。當能源枯竭危機逼近，她的作品成為唯一能喚醒沉睡能源核心的鑰匙。故事融合科技與詩意，刻畫創造與毀滅之間的掙扎，以及藝術如何成為拯救世界的力量。

    請依據以上範例與讀者資訊，生成新的書名與大綱：
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
