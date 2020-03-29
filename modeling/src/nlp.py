import torch
from torch import nn
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
from keras_preprocessing.sequence import pad_sequences


def _get_input_ids(sequences, max_len):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_texts = [tokenizer.tokenize(text) for text in sequences]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    return pad_sequences(
        input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post"
    )

def get_attention_masks(input_ids):
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks


class TextClassifier(nn.Module):
    MAX_LEN = 256

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        hidden_dimension = 32

        if pretrained:
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        else:
            self.bert = DistilBertModel(DistilBertConfig())
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = nn.Linear(self.bert.config.dim, hidden_dimension)
        self.classifier = nn.Linear(hidden_dimension, 1)

    def forward(self, x, attention_mask=None):
        h = self._pre_classify(x, attention_mask=attention_mask)
        h = self.classifier(h)
        return torch.sigmoid(h).squeeze()

    def _device(self):
        return next(self.parameters()).device

    def _pre_classify(self, x, attention_mask=None):
        h = self.bert(input_ids=x, attention_mask=attention_mask)[0]
        sequence_cls = h[:, 0]
        return self.pre_classifier(sequence_cls)

    @classmethod
    def get_convert_ids_fn(cls):
        return _get_input_ids


    def get_sequence_encoding(self, sequence: str) -> torch.tensor:
        sentence = f"[CLS] {sequence} [SEQ]"
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        input_ids = [self.tokenizer.convert_tokens_to_ids(tokenized_sentence)]
        input_ids = pad_sequences(
            input_ids,
            maxlen=self.MAX_LEN,
            dtype="long",
            truncating="post",
            padding="post",
        )
        attention_mask = [[float(i > 0) for i in input_ids[0]]]
        self.eval()
        return self._pre_classify(
            torch.tensor(input_ids).to(self._device()),
            attention_mask=torch.tensor(attention_mask).to(self._device()),
        )

    @classmethod
    def from_file(cls, file_path: str, device: torch.device, **kwargs):
        model = cls()
        model.load_state_dict(torch.load(file_path, map_location=device))
        model.to(device)
        return model
