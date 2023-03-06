import torch.nn as nn
import dataclasses

#TODO BERT
# 1. Embedding
# 2. encoder blocks
# 4. Classifier layer

@dataclasses.dataclass
class BertConfig:
    model_dim: int = 768
    vocab_size: int = 2**15

class Bert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.model_dim
        )

    def forward(self, x):
        return x
