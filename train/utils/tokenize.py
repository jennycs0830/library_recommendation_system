from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt
import numpy as np

def get_sentence_transformer(model_name="shibing624/text2vec-base-chinese"):
    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer
    return model, tokenizer

def chunk_text(text, tokenizer, chunk_size=128):
    tokens = tokenizer.tokenize(text)
    chunks = ["".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks

def count_tokens_after_chunking(text, tokenizer):
    chunks = chunk_text(text, tokenizer)
    return sum(len(tokenizer.tokenize(chunk)) for chunk in chunks)