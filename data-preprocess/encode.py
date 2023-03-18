
import pickle
import os
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
import fire
import numpy as np

def dump_dataset(data, path, sequence_length=512):
    #TODO - use memmap
    num_rows = len(data)
    array = np.zeros((num_rows, sequence_length), dtype=np.uint16)

    for i, row in enumerate(data):
        tokens = row['tokens']
        array[i, :] = tokens

    with open(path, 'wb') as f:
        pickle.dump(array, f)

def save(data, data_test, sequence_length=512, tokenizer_path='./tokenizer.json'):
    pth = os.path.join(os.path.dirname(__file__), 'data.bin')
    pth_test = os.path.join(os.path.dirname(__file__), 'data_test.bin')
    meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')

    meta = {
        'num_rows': len(data),
        'num_rows_test': len(data_test),
        'sequence_length': sequence_length,
        'data_pth': pth,
        'data_pth_test': pth_test,
        'tokenizer_path': tokenizer_path
    }
        
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    dump_dataset(data, pth, sequence_length=sequence_length)
    dump_dataset(data_test, pth_test, sequence_length=sequence_length)

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

    train_data = transformed_data[:int(len(transformed_data)*0.9)]
    test_data = transformed_data[int(len(transformed_data)*0.9):]

    save(
        train_data,
        test_data,
        sequence_length=sequence_length+1,
        tokenizer_path=tokenizer_path
    )

if __name__ == '__main__':
    fire.Fire(main)
