
import pickle
import os
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
import fire
import numpy as np


def save(data, sequence_length=512, tokenizer_path='./tokenizer.json'):
    num_rows = len(data)
    pth = os.path.join(os.path.dirname(__file__), 'data.bin')
    meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')

    #TODO - use memmap
    array = np.zeros((num_rows, sequence_length), dtype=np.uint16)

    meta = {
        'num_rows': num_rows,
        'sequence_length': sequence_length,
        'data_pth': pth,
        'tokenizer_path': tokenizer_path
    }

    for i, row in enumerate(data):
        tokens = row['tokens']
        array[i, :] = tokens

    with open(pth, 'wb') as f:
        pickle.dump(array, f)

    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)


def main(tokenizer_path='./tokenizer.json', sequence_length=128):
    print('Loading tokenizer from {}'.format(tokenizer_path))
    print('sequence_length: {}'.format(sequence_length))
    tokenizer = Tokenizer.from_file(tokenizer_path)
    sequence_length -= 1

    tokenizer.enable_padding(
        pad_id=tokenizer.model.token_to_id('[PAD]'),
        pad_type_id=0,
        length=sequence_length
    )

    data = load_dataset(
        'wikitext',
        'wikitext-2-v1',
        split='train+validation+test'
    )

    cls_id = tokenizer.model.token_to_id('[CLS]')
    pad_id = tokenizer.model.token_to_id('[PAD]')

    data = data.filter(lambda x: len(x['text']) > 0)

    def encode(text):
        enc = tokenizer.encode(text['text'])
        return {"tokens": enc.ids, "len": len(enc)}

    def transform(text):
        tokens = text['tokens']
        length = text['len']
        chunks = []

        if length > sequence_length:
            for i in range(0, length, sequence_length):
                current = tokens[i:i+sequence_length]
                if len(current) == sequence_length:
                    current = [cls_id] + current 
                    chunks.append({"tokens": current, "len": len(current)})
                else:
                    # add cls
                    current = [cls_id] + current
                    current += [pad_id] * (sequence_length - len(current)+1)
                    chunks.append({"tokens": current, "len": len(current)})
        else:
            tokens = [cls_id] + tokens
            chunks.append({"tokens": tokens, "len": length})

        return chunks

    data = data.map(encode, remove_columns=['text'], num_proc=4)

    transformed_data = []
    for row in data:
        transformed_data.extend(transform(row))

    save(transformed_data, sequence_length=sequence_length+1, tokenizer_path=tokenizer_path)

if __name__ == '__main__':
    fire.Fire(main)
