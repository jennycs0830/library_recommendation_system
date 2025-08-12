from fastapi import FastAPI
from pydantic import BaseModel
import torch

from train.models.Clutering import Clustering_Model

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Clustering_Model(embed_dim=768, hidden_dim=256, n_prototype=9)
model.load_state_dict(torch.load("train/best_deepcluster_model.pt", map_location=device))
model.eval()

class EmbeddingRequest(BaseModel):
    input_vector: list[float]

@app.post("/encode")
def  encode(data: EmbeddingRequest):
    try:
        x = torch.tensor(data.input_vector).float().unsqueeze(0)
        with torch.no_grad():
            _, encoded_vector = model(x)
            encoded_vector = encoded_vector.squeeze().tolist()
        
        return {"encoded_vector": encoded_vector}
    except Exception as e:
        return {"error": str(e)}