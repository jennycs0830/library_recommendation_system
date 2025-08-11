import argparse
import os
import pandas as pd
import numpy as np
import torch
from clearml import Dataset, Task, OutputModel
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from utils.utils import plot_embeddings_plotly, process_paragraph_embeddings, get_supervised_dataset
from models.Clutering import Clustering_Model
from models.AutoEncoder import AutoEncoder

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embeds_dataset_name", type=str, default="embeddings_db", help="Name of the dataset to download ")
    parser.add_argument("--all_embeddings_file", type=str, default="all_embeddings.pt", help="Filename of the embeddings dataset")
    parser.add_argument("--df_dataset_name", type=str, default="books_with_intro_cleaned")
    parser.add_argument("--df_file", type=str, default="books_with_intro_cleaned.csv")
    
    parser.add_argument("--autoencoder_file", type=str, help="Filename of the trained autoencoder model")
    parser.add_argument("--input_dim", type=int, default=768, help="Input dimension for the autoencoder")
    parser.add_argument("--hidden_dim1", type=int, default=512, help="First hidden dimension for the autoencoder")
    parser.add_argument("--hidden_dim2", type=int, default=192, help="Second hidden dimension for the autoencoder")
    parser.add_argument("--bottleneck_dim", type=int, default=96, help="Bottleneck dimension for the autoencoder")

    parser.add_argument("--train_unsupervised", action='store_true', help="Flag to train the model in supervised mode")
    parser.add_argument("--supervised_label", type=str, default="category_large_group", help="Label to use for supervised training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--n_prototype", type=int, default=10, help="Number of prototypes for clustering")
    parser.add_argument("--freeze_prototype_niters", type=int, default=10000, help="Number of iterations to freeze prototype weights")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=256)
    
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

def train_DeepCluster(args, logger, model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_loss = float('inf')
    best_weight = None
    losses = []
    for epoch in tqdm(range(args.epochs)):
        all_embeds = []
        with torch.no_grad():
            model.eval()
            for embed in dataloader:
                _, predict_embed = model(embed.to(device))
                predict_embed = F.normalize(predict_embed)
                all_embeds.append(predict_embed.cpu())
            
        all_embeds = torch.cat(all_embeds).numpy()

        kmeans = KMeans(n_clusters=args.n_prototype, n_init=20)
        labels = kmeans.fit_predict(all_embeds)
        labels = torch.LongTensor(labels)
        
        with torch.no_grad():
            centriods = torch.from_numpy(kmeans.cluster_centers_).to(device)
            model.projector.weight.copy_(centriods)

        proto_dataset = TensorDataset(
            dataloader.dataset,
            labels
        )
        proto_loader = DataLoader(proto_dataset, batch_size=64, shuffle=True)
        
        model.train()
        total_loss = 0.0
        for i, (embeds, labels) in enumerate(proto_loader):
            embeds, labels = embeds.to(device), labels.to(device)
            logits, _ = model(embeds)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            if (epoch * len(dataloader) + i) < args.freeze_prototype_niters:
                model.projector.weight.grad = None
                model.projector.weight.bias = None
            optimizer.step()

            total_loss += loss.item()
        
        logger.report_scalar("loss", "train", total_loss, iteration=epoch)
        losses.append(total_loss)
        if scheduler:
            scheduler.step()

        # print(f"Epoch {epoch+1}: Loss: {total_loss:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            best_weight = model.state_dict()

    print(f"Best loss: {best_loss:.4f}")
    logger.report_single_value("best_loss", best_loss)

    return best_loss, best_weight, losses

def train_supervised(args, logger, model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_loss = float('inf')
    best_weight = None
    losses = []

    for epoch in tqdm(range(args.epochs)):
        model.train()
        total_loss = 0.0

        for embeds, labels in dataloader:
            embeds, labels = embeds.to(device), labels.to(device)
            logits, _ = model(embeds)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss)
        logger.report_scalar("loss", "train", total_loss, iteration=epoch)
        if scheduler:
            scheduler.step()

        if total_loss < best_loss:
            best_loss = total_loss
            best_weight = model.state_dict()

    print(f"Best loss: {best_loss:.4f}")
    logger.report_single_value("best_loss", best_loss)

    return best_loss, best_weight, losses


def main():
    args = parse_args()

    task = Task.init(project_name="AI_recommender", task_name="train_DC", task_type=Task.TaskTypes.optimizer)
    task.connect(args)
    logger = task.get_logger()

    all_embeddings, df = load_data(args)
    # plot original embeddings
    plot_embeddings_plotly(all_embeddings, logger=logger, task=task, titles=df['bi_title'], labels=df['category'], method='tsne', title="Original_Embeddings_Category")
    plot_embeddings_plotly(all_embeddings, logger=logger, task=task, titles=df['bi_title'], labels=df['category_large_group'], method='tsne', title="Original_Embeddings_Category_Large_Group")

    # load autoencoder model
    autoencoder = AutoEncoder(args.input_dim, args.hidden_dim1, args.hidden_dim2, args.bottleneck_dim)
    state_dict = torch.load(args.autoencoder_file, map_location='cpu')
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    # process embeddings
    encoded_embeddings = process_paragraph_embeddings(all_embeddings, autoencoder, device='cpu')
    _ = plot_embeddings_plotly(encoded_embeddings, logger=logger, task=task, titles=df['bi_title'], labels=df['category'], method='tsne', title='Emebddings_AutoEncoder-Category')
    _ = plot_embeddings_plotly(encoded_embeddings, logger=logger, task=task, titles=df['bi_title'], labels=df['category_large_group'], method='tsne', title='Emebddings_AutoEncoder_Category')

    if not args.train_unsupervised:
        dataloader, n_prototype = get_supervised_dataset(encoded_embeddings, df, args.supervised_label)
        print(f"Number of prototypes: {n_prototype}")
        model = Clustering_Model(embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, n_prototype=n_prototype)
        model.projector.bias.data.fill_(0)

        _, best_weight, _ = train_supervised(args, logger, model, dataloader)
    else:
        dataloader = DataLoader(encoded_embeddings, batch_size=64, shuffle=False)
        model = Clustering_Model(embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, n_prototype=args.n_prototype)
        model.projector.bias.data.fill_(0)

        _, best_weight, _ = train_DeepCluster(args, logger, model, dataloader)

    # Save the best model
    torch.save(best_weight, "best_deepcluster_model.pt")
    task.upload_artifact(name="best_deepcluster_model", artifact_object="best_deepcluster_model.pt")
    print("Best model saved as 'best_deepcluster_model.pt'")
    output_model = OutputModel(task=task, framework="PyTorch")
    output_model.update_weights(weights_filename="best_deepcluster_model.pt")

    # eval
    model.eval()
    embeddings = torch.tensor(encoded_embeddings).detach().clone()
    with torch.no_grad():
        _, emb = model(embeddings)
    emb = emb.numpy()

    torch.save(emb, "encoded_embeddings.pt")
    emb_dataset = Dataset.create(dataset_project="AI_recommender", dataset_name="encoded_embedding_db")
    emb_dataset.add_files("encoded_embeddings.pt")
    emb_dataset.upload()
    emb_dataset.finalize()

    _ = plot_embeddings_plotly(emb, logger=logger, task=task, titles=df['bi_title'], labels=df['category'], method='tsne', title='Emebddings_AutoEncoder_Clustering_Category')
    _ = plot_embeddings_plotly(emb, logger=logger, task=task, titles=df['bi_title'], labels=df['category_large_group'], method='tsne', title='Emebddings_AutoEncoder_Clustering_Category_Large_Group')

    task.close()

if __name__=="__main__":
    main()