import pickle
from torch.utils.data import Dataset
import numpy as np
from tokenizers import Tokenizer

class WikiText(Dataset):
    def __init__(self, path):

        with open(path, 'rb') as f:
            self.meta = pickle.load(f)
            self.length = self.meta['num_rows']

            tokenizer_path = self.meta['tokenizer_path']
            self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.mask_id = self.tokenizer.model.token_to_id('[MASK]')

        with open(self.meta['data_pth'], 'r+b') as f:
            self.data = np.memmap(
                f,
                dtype=np.uint16,
                mode='r',
                shape=(self.meta['num_rows'], self.meta['sequence_length'])
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx, mask=True):
        ex = self.data[idx].astype(np.int32)
        return torch.tensor(ex, dtype=torch.long)


if __name__ == '__main__': 
    from torch.utils.data import DataLoader
    import torch

    torch.manual_seed(42)
    data = WikiText('./data-preprocess/meta.pkl')

    loader = DataLoader(data, batch_size=1, shuffle=False)
    for i in loader:
        print(i.shape)
        break
