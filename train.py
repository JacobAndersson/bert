from model import Bert, BertConfig
from data import WikiText
from torch.utils.data import DataLoader
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

config = BertConfig(
    model_dim=768,
    vocab_size=2**15,
    n_layers=2,
    n_heads=4,
    ff_dim=3072
)

data = WikiText('./data-preprocess/meta.pkl')
loader = DataLoader(data, batch_size=32, shuffle=True)

model = Bert(config)

print(model)

for (x, y) in loader:
    print(x.shape)
    print(y.shape)
    y_pred = model(x)
    break


