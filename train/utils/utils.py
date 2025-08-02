import matplotlib.pyplot as plt
import numpy as np
import re
import unicodedata
import torch
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from clearml import Logger
import os
import tempfile
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset

def get_supervised_dataset(encoded_embeddings, df, attr):
    label_encoder = LabelEncoder()
    int_labels = label_encoder.fit_transform(df[attr])
    print(f"Unique labels: {label_encoder.classes_}")
    n_prototype = len(np.unique(int_labels))
    print(f"Number of unique labels: {n_prototype}")

    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    if not isinstance(encoded_embeddings, torch.Tensor):
        encoded_embeddings = torch.tensor(encoded_embeddings, dtype=torch.float32)

    int_labels = torch.tensor(int_labels, dtype=torch.long)
    dataset = TensorDataset(encoded_embeddings, int_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    return dataloader, n_prototype

def clean_synopsis(text):
    text = normalize_width(text)
    text = clean_basic(text)
    text = remove_exact_duplicates(text)
    text = remove_MARC(text)

    return text

def remove_exact_duplicates(text):
    sentences = text.split('。')  # Use Chinese period for segmentation
    seen = set()
    unique_sentences = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            unique_sentences.append(s)
            seen.add(s)
    return '。'.join(unique_sentences)

def clean_basic(text):
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = re.sub(r'\s+', ' ', text)     # normalize whitespace
    text = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9.,!?，。！？（）()「」“”\'\"\-:：\n ]+', '', text)  # remove weird symbols
    return text.strip()

def normalize_width(text):
    return unicodedata.normalize('NFKC', text)

def remove_MARC(content):
    arr = content.split("001")
    if len(arr) > 1 and arr[0].strip():
        text = arr[0]
    else:
        text = content
    return text

def token_statistic(logger, token_counts, plot_title):
    mean_tokens = token_counts.mean()
    std_tokens = token_counts.std()
    max_tokens = token_counts.max()
    min_tokens = token_counts.min()

    logger.report_single_value("Original Token Counts - Mean", mean_tokens)
    logger.report_single_value("Original Token Counts - Std", std_tokens)
    logger.report_single_value("Original Token Counts - Max", max_tokens)
    logger.report_single_value("Original Token Counts - Min", min_tokens)

    print(f"Mean number of tokens: {mean_tokens:.2f}")
    print(f"Standard deviation of tokens: {std_tokens:.2f}")
    print(f"Maximum number of tokens: {max_tokens}")
    print(f"Minimum number of tokens: {min_tokens}")

    # Optional: Plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist(token_counts, bins=30, edgecolor='black')
    plt.title(plot_title)
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    
    logger.report_matplotlib_figure(title="Token Histogram", series=plot_title, iteration=0, figure=plt.gcf())
    plt.close()

def chunk_text(text, tokenizer, chunk_size=128):
    tokens = tokenizer.tokenize(text)
    chunks = ["".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks

def count_tokens_after_chunking(text, tokenizer):
    chunks = chunk_text(text, tokenizer)
    return sum(len(tokenizer.tokenize(chunk)) for chunk in chunks)

def plot_embeddings_plotly(
    embeddings,
    titles=None,
    labels=None,
    method='pca',
    title='Interactive Embedding Visualization',
    logger: Logger = None,
    task = None,
    step: int = 0
):
    # Convert to numpy if torch
    if hasattr(embeddings, 'detach'):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Normalize
    embeddings = normalize(embeddings)

    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    
    reduced = reducer.fit_transform(embeddings)

    # Prepare dataframe
    df = pd.DataFrame(reduced, columns=['x', 'y'])
    if titles is not None:
        df['title'] = titles
    if labels is not None:
        df['label'] = labels

    # Plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='label' if labels is not None else None,
        hover_name='title' if titles is not None else None,
        title=title,
        width=1000,
        height=800
    )

    # Save to HTML with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{title.replace(' ', '_')}_{timestamp}.html"
    html_path = os.path.join(tempfile.gettempdir(), filename)
    fig.write_html(html_path)

    print(f"Saved interactive plot to: {html_path}")

    # Upload to ClearML
    if task and logger:
        uploaded_path = task.upload_artifact(artifact_object=html_path, name=title)
        logger.report_text(f"[{title} - Interactive Plot]({uploaded_path})", iteration=step)
        print(f"Uploaded to ClearML: {uploaded_path}")

    return html_path

def chunk_embeddings(embeddings, chunk_size=768):
    return [embeddings[i:i + chunk_size] for i in range(0, len(embeddings), chunk_size)]

def encode_chunks(chunks, model, device='cpu'):
    encoded = []
    for chunk in chunks:
        # print(type(chunk_size))
        if len(chunk) < 768:
            padded = np.pad(chunk, (0, 768 - len(chunk)), mode='constant')
        else:
            padded = chunk

        input_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            encoded_chunk = model.encode(input_tensor).cpu().numpy()
        encoded.append(encoded_chunk.flatten())
        
    return np.concatenate(encoded)

def process_paragraph_embeddings(embedding_array, model, device='cpu'):
    reduced_embeddings = []
    for paragraph_embed in embedding_array:
        chunks = chunk_embeddings(paragraph_embed)
        reduced = encode_chunks(chunks, model, device)
        reduced_embeddings.append(reduced)
    return reduced_embeddings
