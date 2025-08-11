import psycopg2
import pandas as pd
import torch
from clearml import Task, Dataset
import os

conn = psycopg2.connect(
    dbname="library_db",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432",
)

conn.autocommit = True
cur = conn.cursor()

# cur.execute("""CREATE DATABASE library_db;""") # using superuser to create the database in terminal

create_user_table = """
    CREATE TABLE IF NOT EXISTS users (
        user_id SERIAL PRIMARY KEY,
        username VARCHAR(50) NOT NULL UNIQUE,
        user_gender TEXT,
        user_age TEXT,
        user_genres TEXT,
        user_profile TEXT,
        user_embedding FLOAT8[],
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
"""

create_interaction_table = """
    CREATE TABLE IF NOT EXISTS interactions (
        interaction_id SERIAL PRIMARY KEY,
        user_id INT REFERENCES users(user_id),
        book_id INT,
        interaction_type VARCHAR(50),
        interaction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
"""

create_book_table = """
    CREATE TABLE IF NOT EXISTS books (
        book_id SERIAL PRIMARY KEY,
        bi_id VARCHAR(50) NOT NULL,
        isbn VARCHAR(20),
        call_number VARCHAR(50) NOT NULL,
        title TEXT NOT NULL,
        image_url TEXT,
        author TEXT NOT NULL,
        content TEXT NOT NULL,
        publisher TEXT,
        publisher_year TEXT,
        site TEXT,
        category VARCHAR(100),
        category_large_group VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
"""

create_book_embedding_table = """
    CREATE TABLE IF NOT EXISTS book_embeddings (
        book_id INT REFERENCES books(book_id),
        embedding FLOAT8[],
        PRIMARY KEY (book_id)
    );
"""

cur.execute(create_user_table)
print(f"Table users: DONE")

cur.execute(create_interaction_table)
print(f"Table interactions: DONE")

cur.execute(create_book_table)
# Insert initial data into the books table
with open("../data/books_with_intro.csv", "r") as f:
    df = pd.read_csv(f)
    for index, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO books (bi_id, isbn, call_number, title, image_url, author, content, publisher, publisher_year, site, category, category_large_group)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row['bi_id'],
                row['bi_isbn'],
                row['bi_class'],
                row['bi_title'],
                row['bi_image'],
                row['bi_auther'],
                row['bi_content'],
                row['bi_publisher'],
                row['bi_publisher_year'],
                row['bi_site'],
                row['category'],
                row['category_large_group']
            )
        )       
print(f"Table books: DONE")

cur.execute(create_book_embedding_table)
# Insert initial data into the book_embeddings table
# dataset = Dataset.get(dataset_project="AI_recommender", dataset_name="encoded_embedding_db")
# embeddings_path = dataset.get_local_copy()
# print(f"Dataset embeddings_db downloaded to {embeddings_path}")
embeddings = torch.load("../data/encoded_embeddings.pt", weights_only=False)

for book_id, embedding in enumerate(embeddings):
    cur.execute(
        """
        INSERT INTO book_embeddings (book_id, embedding)
        VALUES (%s, %s)
        ON CONFLICT (book_id) DO UPDATE SET embedding = EXCLUDED.embedding;
        """,
        (book_id + 1, embedding.tolist())
    )
print(f"Table book_embedding: DONE")

conn.commit()

cur.close()
conn.close() 
