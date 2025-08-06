import pandas as pd
import os
from datetime import datetime
import streamlit as st
import numpy as np

from components.utils import get_pg_connection, get_book_metadata

def log_interaction(user_id, book_id, interaction_type):
    # Corrected SQL syntax
    add_interaction = """
    INSERT INTO interactions (user_id, book_id, interaction_type)
    VALUES (%s, %s, %s);
    """

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            # Step 1: Insert interaction
            cur.execute(add_interaction, (user_id, book_id, interaction_type))
            print("ACTION: log interaction")

            # Step 2: Get current user embedding
            cur.execute("SELECT user_embedding FROM users WHERE user_id = %s;", (user_id,))
            user_embedding = cur.fetchone()[0]
            print("ACTION: get current user embedding")

            # Step 3: Get book embedding
            cur.execute("SELECT embedding FROM book_embeddings WHERE book_id = %s;", (book_id,))
            book_embedding = cur.fetchone()[0]
            print("ACTION: get current book embedding")

            # Step 4: Convert to numpy arrays
            user_embedding = np.array(user_embedding)
            book_embedding = np.array(book_embedding)

            # Step 5: Update user embedding
            updated_user_embedding = 0.9 * user_embedding + 0.1 * book_embedding

            # Step 6: Save back to DB
            cur.execute(
                "UPDATE users SET user_embedding = %s WHERE user_id = %s;",
                (updated_user_embedding.tolist(), user_id)
            )
            print("ACTION: update user embedding")

        conn.commit()


def display_interaction_history(user_id):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 行為紀錄")

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT book_id, interaction_type, interaction_timestamp
                FROM interactions
                WHERE user_id = %s
                ORDER BY interaction_timestamp DESC
            """, (user_id,))
            interaction_his = cur.fetchall()

            if interaction_his:
                for book_id, interaction_type, interaction_timestamp in interaction_his:
                    book_metadata = get_book_metadata(book_id)[0]
                    title = book_metadata.get("title", "未知書名")
                    category = book_metadata.get("category", "未知分類")
                    time_str = interaction_timestamp.strftime("%Y-%m-%d %H:%M:%S") \
                        if hasattr(interaction_timestamp, "strftime") else str(interaction_timestamp)[:19]
                    
                    st.sidebar.write(f"📘 {title} | {category} | {interaction_type} | 🕒 {time_str}")
            else:
                st.sidebar.info("尚無行為紀錄")
