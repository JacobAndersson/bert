from model import Bert, BertConfig

config = BertConfig(
    model_dim=768,
    vocab_size=2**15
)
model = Bert(config)

print(model)

