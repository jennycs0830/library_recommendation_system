import torch.nn as nn
import torch
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, bottleneck_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, bottleneck_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
        )
    
    def forward(self, input):
        encode =  self.encoder(input)
        decode = self.decoder(encode)

        return decode
    
    def encode(self, input):
        with torch.no_grad():
            return self.encoder(input)
        
class MSE_CosineLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MSE_CosineLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha  # weight for cosine loss

    def forward(self, input, output):
        mse_loss = self.mse(output, input)
        cosine_loss = 1 - F.cosine_similarity(output, input, dim=1).mean()
        return mse_loss + self.alpha * cosine_loss
