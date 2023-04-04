from datasets import load_dataset
from tokenizers import Tokenizer
import numpy as np

import fire

SUBSET = 'cola'

def get_tokenizer(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

def dump_dataset(dataset, path):
    ''' mmap dump dataset to disk '''
    with open(path, 'wb') as f:
        for i, x in enumerate(dataset):
            f.write(np.array(x['tokens'], dtype=np.int32).tobytes())
            f.write(np.array(x['label'], dtype=np.int32).tobytes())
            f.write(np.array(x['length'], dtype=np.int32).tobytes())


def main(tokenizer_path = './tokenizer.json'):
    tokenizer = get_tokenizer(tokenizer_path)
    dataset = load_dataset('glue', SUBSET, split='train')

    cls_token = tokenizer.token_to_id('[CLS]')

    def encode(x):
        tokens = tokenizer.encode(x['sentence']).ids
        tokens = [cls_token] + tokens
        return { "tokens": tokens, 'label': x['label'], 'length': len(tokens) }

    data = dataset.map(encode, batched=False, remove_columns=['sentence'])
    dump_dataset(data, f'./{SUBSET}.bin')

    dataset_test = load_dataset('glue', SUBSET, split='validation')
    data_test = dataset_test.map(encode, batched=False, remove_columns=['sentence'])
    dump_dataset(data_test, f'./{SUBSET}_test.bin')

if __name__ == '__main__':
    fire.Fire(main)
