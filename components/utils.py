import os
import json
import pandas as pd
from datetime import datetime  
import requests
from PIL import Image
from io import BytesIO

DATA_FOLDER = "data"
USERS_DB = os.path.join(DATA_FOLDER, "users.json")
BOOKS_DB = os.path.join(DATA_FOLDER, "books.csv")
INTERACTIONS_DB = os.path.join(DATA_FOLDER, "interactions.csv")
ACTION_WEIGHTS = {
    '瀏覽': 1,
    '收藏': 2, 
    '預約': 3,
    '借閱': 4
}
DEFAULT_IMAGE_PATH = os.path.join(DATA_FOLDER, "default_cover.jpeg")

def load_users():
    if not os.path.exists(USERS_DB):
        return {}
    with open(USERS_DB, "r", encoding='utf-8') as f:
        return json.load(f)
    
def save_users(users):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(USERS_DB, "w", encoding='utf-8') as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

def load_books():
    return pd.read_csv(BOOKS_DB)

def fetch_image_cached(url):
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except:
        return Image.open("data/default_cover.jpeg")