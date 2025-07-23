import pandas as pd
import os
from datetime import datetime
import streamlit as st

ACTION_WEIGHTS = {
    '瀏覽': 1,
    '收藏': 2,
    '預約': 3,
    '借閱': 4
}
DATA_FOLDER = "data"
INTERACTIONS_DB = os.path.join(DATA_FOLDER, "interactions.csv")

def log_interaction(user_id, book, interaction_type):
    log = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "book_id": book["bi_id"],
        "book_title": book["bi_title"],
        'book_category': book.get("category", "—"),
        "action_type": interaction_type,
        "score": ACTION_WEIGHTS.get(interaction_type, 0)
    }
    os.makedirs(DATA_FOLDER, exist_ok=True)
    if not os.path.exists(INTERACTIONS_DB):
        pd.DataFrame([log]).to_csv(INTERACTIONS_DB, index=False)
    else:
        interaction_df = pd.read_csv(INTERACTIONS_DB)
        interaction_df = pd.concat([interaction_df, pd.DataFrame([log])])
        interaction_df.to_csv(INTERACTIONS_DB, index=False)

def display_interaction_history(user_id, interactions_db):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 行為紀錄")
    if os.path.exists(interactions_db):
        interaction_df = pd.read_csv(interactions_db)
        interaction_df['user_id'] = interaction_df['user_id'].astype(str)
        user_df = interaction_df[interaction_df['user_id'] == str(user_id)]
        if not user_df.empty:
            recent = user_df.sort_values("timestamp", ascending=False)
            for _, row in recent.iterrows():
                st.sidebar.write(f"📘 {row['book_title']} | {row['book_category']} | {row['action_type']} | 🕒 {row['timestamp'][:19]}")
        else:
            st.sidebar.info("尚無行為紀錄")
    else:
        st.sidebar.info("尚無行為紀錄")