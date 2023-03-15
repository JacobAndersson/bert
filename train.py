from model import Bert, BertConfig
from data import WikiText
from torch.utils.data import DataLoader
import torch
import numpy as np
import time

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
loader_iter = iter(loader)

def get_batch():
    global loader_iter
    try:
        x, y = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        x, y = next(loader_iter)
    return x, y


model = Bert(config)

lr = 1e-4
max_steps = 100
grad_accumulation_steps = 5

criterion = torch.nn.CrossEntropyLoss(ignore_index=3)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    eps=1e-12
)

def get_lr(step, max_steps, max_lr):
    if step < max_steps/2:
        return 2 * max_lr * step / max_steps
    return 2 * max_lr * (max_steps - step) / max_steps

print(model)

for step in range(max_steps):

    lr = get_lr(step, 100, 1e-4)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    start = time.time()
    for _ in range(grad_accumulation_steps):
        x, y = get_batch()
        y_pred = model(x)
        y_pred = y_pred.transpose(1, 2)

        loss = criterion(y_pred, y)
        loss.backward()

    elapsed = time.time() - start
    print(f"step: {step}, loss: {loss.item():.4f}, lr: {lr:.6f}, batch took: {elapsed:.2f}s")

    optimizer.step()
    optimizer.zero_grad()

    #y_pred = torch.functional.F.softmax(y_pred, dim=1)
    #output_pred = torch.argmax(y_pred, dim=1)
