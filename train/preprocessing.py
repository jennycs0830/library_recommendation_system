from clearml import Task, Dataset
import argparse
import pandas as pd
import numpy as np
import os
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split

from utils.call_number import classify_call_number, classify_call_number_large_group
from utils.utils import token_statistic, count_tokens_after_chunking, clean_synopsis
from utils.tokenize import get_sentence_transformer, chunk_text, count_tokens_after_chunking
from utils.templates import build_book_text, template

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="books_with_intro", help="Name of the dataset to download")

    parser.add_argument("--data_folder", type=str, default="../data", help="Path to the data folder")
    parser.add_argument("--dataset_file", type=str, default="books_with_intro.csv", help="Filename of the dataset")
    parser.add_argument("--selected_cols", type=str, nargs='+', default=["bi_id", "bi_isbn", "bi_class", "bi_title", "bi_image", "bi_content", "bi_auther", "bi_publisher", "bi_publisher_year", "bi_site"], help="Columns to select from the dataset")
    
    parser.add_argument("--tokenizer_model", type=str, default="shibing624/text2vec-base-chinese", help="Model name for the tokenizer")

    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for text processing")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum number of tokens per chunk")
    parser.add_argument("--max_chunks", type=int, default=8, help="Maximum number of chunks per book")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    task = Task.init(project_name="AI_recommender", task_name="Dataset preprocessing", task_type=Task.TaskTypes.optimizer)
    task.connect(args)
    logger = task.get_logger()

    dataset = Dataset.get(dataset_name=args.dataset_name, dataset_project="AI_recommender")
    path = dataset.get_local_copy()
    df = pd.read_csv(os.path.join(path, args.dataset_file), header=0) # header 1
    print(f"Dataset {args.dataset_name} downloaded to {path}")
    print(f"Dataset shape: {df.shape}")

    # select only the specified columns
    df = df[args.selected_cols]

    # drop incomplete rows
    df.replace(to_replace=["NULL", "null", "N/A", "na", "NaN", ""], value=np.nan, inplace=True)
    df = df.dropna(subset=args.selected_cols).reset_index(drop=True)
    print(f"Dataset shape after dropping incomplete rows: {df.shape}")

    # call_number to category mapping
    df['category'] = df['bi_class'].apply(classify_call_number)
    df['category_large_group'] = df['bi_class'].apply(classify_call_number_large_group)
    
    # get model and tokenizer
    model, tokenizer = get_sentence_transformer(args.tokenizer_model)

    # statistics on token counts - Original
    token_counts_original = df['bi_content'].apply(lambda x: count_tokens_after_chunking(x, tokenizer))
    token_statistic(logger, token_counts_original, plot_title="Token Counts Original")

    # data cleaning
    cleaned_content = df['bi_content'].apply(clean_synopsis)

    # statistics on token counts - After cleaning
    token_counts_cleaned = cleaned_content.apply(lambda x: count_tokens_after_chunking(x, tokenizer))
    token_statistic(logger, token_counts_cleaned, plot_title="Token Counts After Cleaning")

    # save and upload cleaned dataset
    df['bi_content'] = cleaned_content
    cleaned_dataset_path = f"{args.data_folder}/books_with_intro_cleaned.csv"
    df.to_csv(cleaned_dataset_path, index=False)
    print(f"Cleaned dataset saved to {cleaned_dataset_path}")

    # upload cleaned dataset to ClearML
    cleaned_dataset = Dataset.create(dataset_name=args.dataset_name + "_cleaned", dataset_project="AI_recommender")
    cleaned_dataset.add_files(cleaned_dataset_path)
    cleaned_dataset.upload()
    cleaned_dataset.finalize()
    print(f"Cleaned dataset uploaded to ClearML with ID: {cleaned_dataset.id}")

    # get embeddings
    logger.report_text(template, "template")
    book_texts = df.apply(build_book_text, axis=1).tolist()
    all_chunked_texts = []

    for text in book_texts:
        chunks = chunk_text(text, tokenizer, args.chunk_size)
        all_chunked_texts.append(chunks)
    
    final_embeddings = []
    for chunks in tqdm(all_chunked_texts):
        selected = chunks[:args.max_chunks]
        while len(selected) < args.max_chunks:
            selected.append("")
        embeddings = model.encode(selected, convert_to_tensor=True)
        final_embeddings.append(torch.flatten(embeddings))

    all_embeddings = torch.stack(final_embeddings)
    train_tensor, test_tensor = train_test_split(all_embeddings, test_size=0.2)

    torch.save(all_embeddings, "all_embeddings.pt")
    torch.save(train_tensor, "train_embeddings.pt")
    torch.save(test_tensor, "test_embeddings.pt")
    
    embeddings_dataset = Dataset.create(dataset_name="embeddings_db", dataset_project="AI_recommender")
    embeddings_dataset.add_files("all_embeddings.pt")
    embeddings_dataset.add_files("train_embeddings.pt")
    embeddings_dataset.add_files("test_embeddings.pt")
    embeddings_dataset.upload()
    embeddings_dataset.finalize()
    print(f"Embeddings dataset uploaded to ClearML with ID: {embeddings_dataset.id}")
    
    os.remove("all_embeddings.pt")
    os.remove("train_embeddings.pt")
    os.remove("test_embeddings.pt")
    
    # log the dataset
    task.close()

if __name__ == "__main__":
    main()