import numpy as np
import torch
from models.vit import ViT



model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=2,
    dim = 1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1,
)

img = torch.randn(1, 3, 256, 256)

preds = model(img).cpu().detach().numpy()

print(preds)
