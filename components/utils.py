import os
import json
import pandas as pd
from datetime import datetime  
import requests
from PIL import Image
from io import BytesIO
import requests
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATA_FOLDER = "data"
DEFAULT_IMAGE_PATH = os.path.join(DATA_FOLDER, "default_cover.jpeg")

def get_book_metadata(book_ids):
    try:
        placeholders = ','.join(['%s'] * len(book_ids))
    except:
        placeholders = book_ids
        
    query = f"SELECT book_id, isbn, call_number, title, image_url, author, content, publisher, publisher_year, site, category, category_large_group FROM books WHERE book_id IN ({placeholders})"
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, book_ids)
            rows = cur.fetchall()
            # print(rows)
            # print(len(rows))
            return [
                {"book_id": r[0], 
                 "isbn": r[1],
                 "call_number": r[2],
                 "title": r[3], 
                 "image_url": r[4], 
                 "author": r[5],
                 "content": r[6], 
                 "publisher": r[7],
                 "publisher_year": r[8],
                 "site": r[9],
                 "category": r[10],
                 "category_large_group": r[11]}
                for r in rows
            ]
        
def get_pg_connection():
    return psycopg2.connect(
        dbname="library_db",
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

def fetch_image_cached(url):
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except:
        return Image.open("data/default_cover.jpeg")

def encode_user_embedding(original_embedding, autoencoder_url="http://localhost:8002/encode", clustering_url="http://localhost:8003/encode"):
    target_dim = 768
    if len(original_embedding) < target_dim:
        repeat_times = target_dim // len(original_embedding)
        remainder = target_dim % len(original_embedding)
        original_embedding = original_embedding * repeat_times + original_embedding[:remainder]
    elif len(original_embedding) > target_dim:
        original_embedding = original_embedding[:target_dim]

    response = requests.post(autoencoder_url, json={"input_vector": original_embedding})
    response.raise_for_status()
    autoencoder_embeds = response.json()["encoded_vector"]
    print("Autoencoder encoded: SUCCESS")
    # autoencoder_embeds = original_embedding
    
    if len(autoencoder_embeds) < target_dim:
        repeat_times = target_dim // len(autoencoder_embeds)
        remainder = target_dim % len(autoencoder_embeds)
        autoencoder_embeds = autoencoder_embeds * repeat_times + autoencoder_embeds[:remainder]
    elif len(autoencoder_embeds) > target_dim:
        autoencoder_embeds = autoencoder_embeds[:target_dim]
    # autoencoder_embeds += [0.0] * padding_length
    response = requests.post(clustering_url, json={"input_vector": autoencoder_embeds})
    response.raise_for_status()
    clustering_embeds = response.json()["encoded_vector"]
    print("Clustering encoded: SUCCESS")

    return clustering_embeds