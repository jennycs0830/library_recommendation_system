# filepath: /library_recommendation_system/src/utils/model.py
import torch
from sentence_transformers import SentenceTransformer
import faiss
import os

EMBEDDING_MODEL_NAME = 'shibing624/text2vec-base-multilingual'
CLUSTERING_MODEL_PATH = "deepcluster_model.pt"
INDEX_DB = "book_index.faiss"
DATA_FOLDER = "data"

def get_embedding_model():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Embedding model {EMBEDDING_MODEL_NAME} loaded.")
    return model

def get_faiss_index():
    index = faiss.read_index(os.path.join(DATA_FOLDER, INDEX_DB))
    print(f"Faiss index DB {INDEX_DB} loaded.")
    return index

def get_clustering_model():
    model = torch.load(CLUSTERING_MODEL_PATH, weights_only=False)
    model.eval()
    print(f"Clustering model {CLUSTERING_MODEL_PATH} loaded.")
    return model

def load_models():
    embedding_model = get_embedding_model()
    faiss_index = get_faiss_index()
    clustering_model = get_clustering_model()
    return embedding_model, faiss_index, clustering_model