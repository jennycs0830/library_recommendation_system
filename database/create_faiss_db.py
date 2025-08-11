import psycopg2
import streamlit as st
import numpy as np
import faiss

def connect_db():
    conn = psycopg2.connect(
        dbname="library_db",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    return cur, conn

def fetch_embeddings():
    cur, conn = connect_db()

    cur.execute("SELECT book_id, embedding FROM book_embeddings ORDER BY book_id")
    rows = cur.fetchall()

    book_ids = []
    embeddings = []

    for book_id, emb in rows:
        book_ids.append(book_id)
        embeddings.append(np.array(emb, dtype=np.float32))

    cur.close()
    conn.close()

    return book_ids, np.vstack(embeddings)

def get_faiss_index(embeddings, dim):
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, book_ids, index_path="faiss_books.index", id_map_path="book_ids.npy"):
    faiss.write_index(index, index_path)
    np.save(id_map_path, book_ids)
    print(f"Index saved to {index_path}, ID map saved to {id_map_path}")

def create_faiss_db():
    book_ids, embeddings = fetch_embeddings()
    print(f"Loaded {len(book_ids)} embeddings with shape {embeddings.shape}")

    dim = embeddings.shape[1]
    index = get_faiss_index(embeddings.astype('float32'), dim)
    save_index(index, book_ids)

def main():
    create_faiss_db()

if __name__=="__main__":
    main()