import torch
import torch.nn as nn

class Clustering_Model(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_prototype=100):
        super(Clustering_Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.projector = nn.Linear(hidden_dim, n_prototype)
    
    def forward(self, sentence_embeds, project_grad=True):
        embeds = self.encoder(sentence_embeds)
        if project_grad:
            logits = self.projector(embeds)
        else:
            with torch.no_grad():
                logits = self.projector(embeds)
        return logits, embeds