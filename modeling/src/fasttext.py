import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from keras_preprocessing.sequence import pad_sequences
from torchtext.data.utils import get_tokenizer


def _tokens_to_ids(tokens, vocab):
    return [vocab.stoi[word] for word in tokens if word in vocab.stoi]


def _get_fasttext_ids(sequences, max_len):
    tokenize = get_tokenizer("basic_english")
    vocab = FastTextClassifier.vocab
    tokenized_texts = [tokenize(text) for text in sequences]

    input_ids = [_tokens_to_ids(tokens, vocab) for tokens in tokenized_texts]
    return pad_sequences(
        input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post"
    )


class FastTextClassifier(nn.Module):
    MAX_LEN = 256
    vocab = torchtext.vocab.FastText()

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        bag_hidden_dimension = 128
        fc_hidden_dimension = 64

        if pretrained:
            bag_hidden_dimension = self.vocab.dim
            self.embedding = nn.EmbeddingBag.from_pretrained(self.vocab.vectors)
        else:
            self.embedding = nn.EmbeddingBag(
                len(self.vocab), bag_hidden_dimension, sparse=False
            )
        self.pre_classifier = nn.Linear(bag_hidden_dimension, fc_hidden_dimension)
        self.classifier = nn.Linear(fc_hidden_dimension, 1)

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x, attention_mask):
        text = x.masked_select(attention_mask.bool())
        lengths = [0] + attention_mask.long().sum(dim=1).tolist()
        offsets = torch.tensor(lengths[:-1]).cumsum(dim=0).to(self._device())

        embedded = self.embedding(text, offsets)
        pre_classified = self.pre_classifier(embedded)
        h = self.classifier(pre_classified)
        return torch.sigmoid(h).squeeze()

    def _device(self):
        return next(self.parameters()).device

    @classmethod
    def get_convert_ids_fn(cls):
        return _get_fasttext_ids

    @classmethod
    def from_file(cls, file_path: str, device: torch.device, pretrained: bool):
        model = cls(pretrained=pretrained)
        model.load_state_dict(torch.load(file_path, map_location=device))
        model.to(device)
        return model
