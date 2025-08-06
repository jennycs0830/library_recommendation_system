from fastapi import FastAPI
from pydantic import BaseModel
import torch

from train.models.AutoEncoder import AutoEncoder

app = FastAPI()

model = AutoEncoder(input_dim=768, hidden_dim1=512, hidden_dim2=192, bottleneck_dim=96)
model.load_state_dict(torch.load("train/best_autoencoder_20250731_210546.pt"))
model.eval()

class EmbeddingRequest(BaseModel):
    input_vector: list[float]

@app.post("/encode")
def  encode(data: EmbeddingRequest):
    try:
        x = torch.tensor(data.input_vector).float().unsqueeze(0)
        with torch.no_grad():
            encoded_vector = model.encode(x).squeeze().tolist()
        return {"encoded_vector": encoded_vector}
    except Exception as e:
        return {"error": str(e)}