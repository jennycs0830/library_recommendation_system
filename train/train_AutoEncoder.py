import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from clearml import Task, Dataset, OutputModel
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from models.AutoEncoder import AutoEncoder

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default="embeddings_db", help="Name of the dataset to download")
    parser.add_argument("--all_embeddings_file", type=str, default="all_embeddings.pt", help="Filename of the embeddings dataset")
    parser.add_argument("--training_file", type=str, default="train_embeddings.pt", help="Filename of the training dataset")
    parser.add_argument("--testing_file", type=str, default="test_embeddings.pt", help="Filename of the testing dataset")
    
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch sizeq for training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--input_dim', type=int, default=768, help="Input dimension for the model")
    parser.add_argument('--hidden_dim1', type=int, default=512, help="First hidden dimension for the model")
    parser.add_argument('--hidden_dim2', type=int, default=192, help="Second hidden dimension for the model")
    parser.add_argument('--bottleneck_dim', type=int, default=96, help="Bottleneck dimension for the model")                        

    parser.add_argument("--ckpt_folder", type=str, default="checkpoints", help="Folder to save model checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")

    args = parser.parse_args()
    return args

def split_data(args, embeds):
    input_dim = args.input_dim
    chunks = []

    for arr in embeds:
        t = torch.as_tensor(arr, dtype=torch.float32)
        t = t.flatten()  # ensure 1D
        remainder = t.numel() % input_dim
        if remainder != 0:
            pad_len = input_dim - remainder
            t = torch.cat([t, t[:pad_len]], dim=0)
        num_chunks = t.numel() // input_dim
        t = t.view(num_chunks, input_dim)
        chunks.append(t)

    if len(chunks) == 0:
        return torch.empty(0, input_dim)

    return torch.cat(chunks, dim=0) 

def load_data(args, path):
    all_embeddings = torch.load(os.path.join(path, args.all_embeddings_file))

    train_embeddings = torch.load(os.path.join(path, args.training_file))
    train_embeddings = split_data(args, train_embeddings)
    print(f"Len of training data: {len(train_embeddings)}")
    train_embeddings = torch.tensor(train_embeddings)
    train_set = TensorDataset(train_embeddings)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    test_embeddings = torch.load(os.path.join(path, args.testing_file))
    test_embeddings = split_data(args, test_embeddings)
    train_embeddings = torch.tensor(test_embeddings)
    test_set = TensorDataset(test_embeddings)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)  

    return all_embeddings, train_embeddings, train_loader, test_embeddings, test_loader

def train_autoencoder(args, logger, model, train_loader, test_loader, optimizer, scheduler,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    os.makedirs(args.ckpt_folder, exist_ok=True)

    best_losses = float('inf')
    best_weights = None
    criterion = nn.MSELoss()
    start_epoch = 0

    if args.resume:
        checkpoints = sorted(
            [f for f in os.listdir(args.ckpt_folder) if f.startswith("epoch_") and f.endswith(".pt")],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        if checkpoints:
            latest_ckpt = os.path.join(args.ckpt_folder, checkpoints[-1])
            checkpoint = torch.load(latest_ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_losses = checkpoint['best_loss']
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from checkpoint: {latest_ckpt} (epoch {start_epoch})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        for (x_batch, ) in train_loader:
            x_batch = x_batch.to(device)
            x_batch = F.normalize(x_batch, p=2, dim=1)  # Normalize the input
            
            decoded = model(x_batch)
            decoded = F.normalize(decoded, p=2, dim=1)  # Normalize the output
            
            loss = criterion(decoded, x_batch) # target for cosine loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (x_batch, ) in test_loader:
                x_batch = x_batch.to(device)
                x_batch = F.normalize(x_batch, p=2, dim=1)  # Normalize the input

                target = torch.ones(x_batch.size(0)).to(device)
                decoded = model(x_batch)
                decoded = F.normalize(decoded, p=2, dim=1)  # Normalize the output
                
                loss = criterion(decoded, x_batch) # target for cosine loss
                # loss = criterion(decoded, x_batch, target) # target for cosine loss
                test_loss += loss.item()

        if scheduler:
            scheduler.step()

        logger.report_scalar("train_loss", "loss", iteration=epoch, value=train_loss / len(train_loader))
        logger.report_scalar("test_loss", "loss", iteration=epoch, value=test_loss / len(test_loader))
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        if best_losses > test_loss:
            best_losses = test_loss
            best_weights = model.state_dict()

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.ckpt_folder, f"epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_loss': best_losses
            }, ckpt_path)
            print(f"Checkpoint saved at: {ckpt_path}")
            
    print(f"Best loss: {best_losses}")
    return best_losses, best_weights
 
def main():
    args = parse_args()
    task = Task.init(project_name="AI_recommender", task_name="Train AutoEncoder", task_type=Task.TaskTypes.optimizer)
    task.connect(args)
    logger = task.get_logger()

    dataset = Dataset.get(dataset_name=args.dataset_name, dataset_project="AI_recommender")
    path = dataset.get_local_copy()
    print(f"Dataset {args.dataset_name} downloaded to {path}")

    all_embeddings, train_embeddings, train_loader, test_embeddings, test_loader = load_data(args, path)

    model = AutoEncoder(
        input_dim=args.input_dim,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        bottleneck_dim=args.bottleneck_dim
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    best_losses, best_weights = train_autoencoder(
        args, logger, model, train_loader, test_loader, optimizer, scheduler
    )
    logger.report_single_value("best_loss", best_losses)
    model.load_state_dict(best_weights)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"best_autoencoder_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved to {model_path}")

    task.upload_artifact(name=model_path, artifact_object=model_path)
    output_model = OutputModel(task=task, framework="PyTorch")
    output_model.update_weights(weights_filename=model_path)

    task.close()

if __name__ == "__main__":
    main()