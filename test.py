from datasets import load_dataset
from tokenizers import Tokenizer
from model import Bert, BertConfig
import torch
import time
from sklearn.metrics import  matthews_corrcoef

import fire

SUBSET = 'cola'
SEQ_LEN = 128

def get_tokenizer(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

def load_and_pre_dataset(tokenizer, split='train'):
    dataset = load_dataset('glue', SUBSET, split=split)
    cls_token = tokenizer.token_to_id('[CLS]')
    pad_token = tokenizer.token_to_id('[PAD]')

    def encode(x):
        tokens = tokenizer.encode(x['sentence']).ids
        tokens = [cls_token] + tokens
        if (len(tokens) < SEQ_LEN):
            tokens += [pad_token] * (SEQ_LEN - len(tokens))
        return { "tokens": tokens, 'label': x['label'], 'length': len(tokens) }

    data = dataset.map(encode, batched=False, remove_columns=['sentence'])
    data.set_format(type='torch', columns=['tokens', 'label'])

    return data

def predict(model, data, device='cuda'):
    predictions = []
    all_labels = []
    total_correct = 0
    total = 0

    for batch in data:
        with torch.no_grad():
            input_seq = batch['tokens'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_seq)
            full_preds = torch.argmax(logits, dim=2)
            cls_pred = full_preds[:, 0]
            predictions.extend(cls_pred.tolist())
            num_correct = torch.sum(cls_pred == labels)
            all_labels.extend(labels.tolist())

            total_correct += num_correct
            total += len(labels)

    print(matthews_corrcoef(all_labels, predictions))
    return predictions, (total_correct / total).item()

def replace_classifier(model, num_classes):
    model.output = torch.nn.Linear(model.config.model_dim, num_classes)
    return model

def main(
    tokenizer_path = './tokenizer.json',
    model_dim=768,
    vocab_size=2**15,
    n_layers=2,
    n_heads=4,
    ff_dim=3072,
    batch_size=32,
    device='cuda',
    bias=False,
    num_classes=2,
    model_path=None
):
    tokenizer = get_tokenizer(tokenizer_path)

    data_train = load_and_pre_dataset(tokenizer)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size)

    data_test = load_and_pre_dataset(tokenizer, split='validation')
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size)

    config = BertConfig(
        model_dim=model_dim,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        bias=bias,
    )

    model = Bert(config)

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

    if model_path:
        print('loading model from', model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

    model = replace_classifier(model, num_classes)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-6,
        betas=(0.9, 0.98),
        weight_decay=0.01,
        eps=1e-12
    )

    for epoch in range(10):
        print('epoch', epoch)
        for i, row in enumerate(dataloader_train):
            x = row['tokens']
            y = row['label']
            x = x.to(device)
            y = y.to(device)

            with ctx:
                optimizer.zero_grad()
                y_pred = model(x)
                y_pred = y_pred.transpose(1, 2)

                cls_pred = y_pred[:, :, 0] 

                loss = criterion(cls_pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if (i % 10 == 0):
                predictions, accuracy = predict(model, dataloader_test, device)
                print('accuracy', accuracy)

if __name__ == '__main__':
    fire.Fire(main)
