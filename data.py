import pickle
from torch.utils.data import Dataset
import numpy as np
from tokenizers import Tokenizer
import math
import torch

class WikiText(Dataset):
    def __init__(self, path, train=True):

        with open(path, 'rb') as f:
            self.meta = pickle.load(f)

            tokenizer_path = self.meta['tokenizer_path']
            self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.mask_id = self.tokenizer.model.token_to_id('[MASK]')
        self.pad_id = self.tokenizer.model.token_to_id('[PAD]')

        self.vocab_size = self.tokenizer.get_vocab_size()

        data_pth = self.meta['data_pth'] if train else self.meta['data_pth_test']
        num_rows = self.meta['num_rows'] if train else self.meta['num_rows_test']

        self.length = num_rows
        print('Loading data from {}'.format(data_pth))

        with open(data_pth, 'r+b') as f:
            self.data = np.memmap(
                f,
                dtype=np.uint16,
                mode='r',
                shape=(num_rows, self.meta['sequence_length']),
                order='F'
            )

    def __len__(self):
        return self.length

    def __getitem__(self, i, mask=True):
        ex = self.data[i].astype(np.int32)

        y = []

        for idx, token in enumerate(ex):
            prob = np.random.random()

            if prob < 0.15:
                prob /= 0.15

                replace_token = token
                if prob < 0.8:
                    replace_token = self.mask_id
                elif prob < 0.9:
                    replace_token = np.random.randint(0, self.vocab_size)

                y.append(replace_token)
            else:
                y.append(self.pad_id)

        y = torch.tensor(y, dtype=torch.long)
        return ex, y

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    torch.manual_seed(42)
    np.random.seed(42)
    data = WikiText('./data-preprocess/meta.pkl')

    loader = DataLoader(data, batch_size=1, shuffle=False)
    for (x, y) in loader:
        print(x)
        print(y)
        break
