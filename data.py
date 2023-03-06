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
            print(self.tokenizer)
            
        print(self.tokenizer.token_to_id('[CLS]'))
         

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
        ex = self.data[idx].astype(np.int16)
        #add bos and eos
        #ex = np.insert(ex, 0, self.bos_id)
        #ex = np.append(ex, self.eos_id)

        return ex


if __name__ == '__main__': 
    from torch.utils.data import DataLoader
    data = WikiText('./data-preprocess/meta.pkl')

    loader = DataLoader(data, batch_size=64, shuffle=True)
    for i in loader:
        print(i)
        print(i.shape)
        break
