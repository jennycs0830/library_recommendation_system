import argparse
import os
import pandas as pd
import numpy as np
import torch
from clearml import Dataset, Task

from utils.utils import plot_embeddings_plotly

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embeds_dataset_name", type=str, default="embeddings_db", help="Name of the dataset to download ")
    parser.add_argument("--all_embeddings_file", type=str, default="all_embeddings.pt", help="Filename of the embeddings dataset")
    parser.add_argument("--df_dataset_name", type=str, default="books_with_intro")
    parser.add_argument("--df_file", type=str, default="books_with_intro_cleaned.csv")
    
    parser.add_argument("--autoencoder_file", type=str, help="Filename of the trained autoencoder model")
    
    args = parser.parse_args()

    return args

def load_data(args):
    embeds_dataset = Dataset.get(dataset_name=args.embeds_dataset_name, dataset_project="AI_recommender")
    embeds_path = embeds_dataset.get_local_copy()
    print(f"Dataset {args.embeds_dataset_name} downloaded to {embeds_path}")
    all_embeddings = torch.load(os.path.join(embeds_path, args.all_embeddings_file))

    df_dataset = Dataset.get(dataset_name=args.df_dataset_name, dataset_project="AI_recommender")
    df_path = df_dataset.get_local_copy()
    print(f"Dataset {args.df_dataset_name} downloaded to {df_path}")
    df = pd.read_csv(os.path.join(df_path, args.df_file))
    
    return all_embeddings, df

def main():
    args = parse_args()

    task = Task.init(project_name="AI_recommender", task_name="train_DC", type=Task.TaskTypes.optimizer)
    task.connect(args)
    logger = task.get_logger()

    all_embeddings, df = load_data(args)

    # plot original embeddings
    plot_embeddings_plotly(all_embeddings, logger=logger, titles=df['bi_title'], labels=df['category'], method='tsne')
    plot_embeddings_plotly(all_embeddings, logger=logger, titles=df['bi_title'], labels=df['category_large_group'], method='tsne')



