import os
import numpy as np
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
import fire

def save(data, split):
    data_length = sum(map(lambda x: x['len'], data))
    pth = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    array_file = np.memmap(pth, dtype=np.uint16, mode="w+", shape=(data_length,))

    counter = 0
    for row in data:
        tokens = row['tokens']
        array_file[counter:counter+row['len']] = tokens
        counter += row['len']

    array_file.flush()

def main(tokenizer_path = 'tokenizer.json', min_length = 0, sequence_length = 512):
    tokenizer = Tokenizer.from_file(tokenizer_path)

    data = load_dataset('wikitext', 'wikitext-2-v1', split='train+validation+test')
    data = data.filter(lambda x: len(x['text']) > 0)

    def encode(text):
        enc = tokenizer.encode(text['text'])
        return { "tokens": enc.ids, "len": len(enc)}

    def transform(text):
        # everythin over sequence_length i chunked into multiple sequences
        # texts is the encoded output from encode()

        tokens = text['tokens']
        length = text['len']
        chunks = []

        if length > sequence_length:
            for i in range(0, length, sequence_length):
                current = tokens[i:i+sequence_length]
                chunks.append({ "tokens": current, "len": len(current) })
        else:
            chunks.append({ "tokens": tokens, "len": length })

        return chunks

    data = data.map(encode, remove_columns=['text'], num_proc=4)

    if min_length > 0:
        data = data.filter(lambda x: x['len'] > min_length)

    transformed_data = []
    for row in data:
        transformed_data.extend(transform(row))

    save(transformed_data, 'train')

if __name__ == '__main__':
    fire.Fire(main)
