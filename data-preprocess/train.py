from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer
from tokenizers.processors import TemplateProcessing

import dataset

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=2**15
)

tokenizer.pre_tokenizer = BertPreTokenizer()
data = dataset.load()

def train_iter(batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]['text']

tokenizer.train_from_iterator(train_iter(1000), trainer=trainer, length=len(data))

tokenizer.save("./tokenizer.json")
