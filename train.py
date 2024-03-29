from model import Bert, BertConfig
from data import WikiText
from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import fire
import wandb
import logging

FORMAT = '%(asctime)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

torch.manual_seed(42)
np.random.seed(42)
torch.set_printoptions(threshold=10_000)

logger = logging.getLogger(__name__)


def main(
    model_dim=768,
    vocab_size=2**15,
    n_layers=2,
    n_heads=4,
    ff_dim=3072,
    max_lr=1e-3,
    max_steps=100,
    grad_accumulation_steps=5,
    batch_size=32,
    device='cuda',
    dry_run=False,
    bias=False,
    use_wandb=False,
    checkpoint_interval=100,
    testing_interval=100,
    testing_samples=20,
    data_path='./data-preprocess/meta.pkl',
    dropout=0.1,
):
    config = BertConfig(
        model_dim=model_dim,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        bias=bias,
        dropout=dropout,
    )

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

    if use_wandb:
        logger.info('Init wandb')
        wandb.init(project='bert-cram', config=config)

    data = WikiText(data_path, train=True)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    loader_iter = iter(loader)

    if len(loader) // grad_accumulation_steps < max_steps:
        logger.info('Fewer steps than updates. Model will see the same data multiple times.')

    data_test = WikiText(data_path, train=False)
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)
    loader_test_iter = iter(loader_test)

    def get_batch(train, device=device):
        nonlocal loader_iter
        nonlocal loader_test_iter
        nonlocal loader
        nonlocal loader_test

        data_loader = loader_iter if train else loader_test_iter
        try:
            x, y = next(data_loader)
        except StopIteration:
            if train:
                loader_iter = iter(loader)
                data_loader = loader_iter
            else:
                loader_test_iter = iter(loader_test)
                data_loader = loader_test_iter

            x, y = next(data_loader)

        x = x.to(device)
        y = y.to(device)
        return x, y

    model = Bert(config)
    model = torch.compile(model)
    model.to(device)

    pad_idx = 3
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.98),
        weight_decay=0.01,
        eps=1e-12
    )

    def get_lr(step, max_steps, max_lr):
        if step < max_steps/2:
            return 2 * max_lr * step / max_steps
        return 2 * max_lr * (max_steps - step) / max_steps

    logger.info(model)
    logger.info(config)
    param_count = model.parameter_count()
    logger.info(f'Parameter count {param_count/1e6:.2f}M')

    best_loss = float('inf')

    for step in range(max_steps):

        lr = get_lr(step, max_steps, max_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        start = time.time()
        for _ in range(grad_accumulation_steps):
            x, y = get_batch(True)

            with ctx:
                y_pred = model(x)
                y_pred = y_pred.transpose(1, 2)

            loss = criterion(y_pred, y)
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        elapsed = time.time() - start
        logger.info(f"step: {step}, loss: {loss.item():.4f}, lr: {lr}, batch took: {elapsed:.2f}s")

        if use_wandb:
            wandb.log({'loss': loss.item(), 'lr': lr})

        if step % checkpoint_interval == 0:
            print('Saving checkpoint')
            checkpoint = {
                'model': model.state_dict(),
                'step': step,
                'optimizer': optimizer.state_dict(),
                'lr': lr,
                'config': config,
                'loss': loss.item(),
            }

            torch.save(checkpoint, f'checkpoints/{step}.pt')

        if step % testing_interval == 0:
            logger.info(f'Testing {step}')

            avg_loss = 0
            with torch.no_grad():
                for i in range(testing_samples):
                    x, y = get_batch(False)
                    with ctx:
                        y_pred = model(x)
                        y_pred = y_pred.transpose(1, 2)

                    loss = criterion(y_pred, y)
                    avg_loss += loss.item()

            avg_loss /= testing_samples

            logger.info(f'Test loss: {avg_loss}')

            if avg_loss < best_loss:
                best_loss = avg_loss

                logger.info('New best found. Saving model')
                checkpoint = {
                    'model': model.state_dict(),
                    'step': step,
                    'optimizer': optimizer.state_dict(),
                    'lr': lr,
                    'config': config,
                    'loss': loss.item(),
                }

                torch.save(checkpoint, f'checkpoints/best.pt')

            if use_wandb:
                wandb.log({'test_loss': avg_loss})

        if dry_run and step >= 5:
            break

if __name__ == '__main__':
    fire.Fire(main)
