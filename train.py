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
loader = DataLoader(data, batch_size=2, shuffle=True)

model = Bert(config)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    eps=1e-12
)

print(model)

for batch_idx, (x, y) in enumerate(loader):
    print('x', x.shape)
    print('y', y.shape)

    for _ in range(10):
        y_pred = model(x)
        print('y_pred', y_pred.shape)
        y_pred = y_pred.transpose(1, 2)
       
        loss = criterion(y_pred, y)
        print('loss', loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    y_pred = torch.functional.F.softmax(y_pred, dim=1)
    output_pred = torch.argmax(y_pred, dim=1)
    print('pred', output_pred)
    break

    if batch_idx == 5:
        break
