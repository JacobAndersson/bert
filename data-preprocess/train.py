
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer
from tokenizers.processors import TemplateProcessing

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=2**15
)

tokenizer.pre_tokenizer = BertPreTokenizer()


'''
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $0",
    special_tokens=[("[CLS]", 1)]
)
'''

data = load_dataset('wikitext', 'wikitext-2-v1', split='train+validation+test')

data = data.filter(lambda x: len(x['text']) > 0)

def train_iter(batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]['text']

tokenizer.train_from_iterator(train_iter(1000), trainer=trainer, length=len(data))

tokenizer.save("./tokenizer.json")
