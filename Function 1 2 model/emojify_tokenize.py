import torch


class EmojifyTokenizer():
    def __init__(self, tokenizer, sentences, labels):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.labels = labels
        self.input_ids = []
        self.attention_masks = []
        self.max_len = self.calculate_max_len()

    def tokenize(self):
        self.input_ids = []
        self.attention_masks = []
        for sentence in self.sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,  # insert '[CLS]' and '[SEP]'
                max_length=64,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )

            self.input_ids.append(encoded_dict['input_ids'])
            self.attention_masks.append(encoded_dict['attention_mask'])

    def get_tensor(self):
        input_ids = torch.cat(self.input_ids, dim=0)
        attention_masks = torch.cat(self.attention_masks, dim=0)
        labels = torch.tensor(self.labels)
        return input_ids, attention_masks, labels

    def calculate_max_len(self):
        self.max_len = 0
        for sentence in self.sentences:
            input_ids = self.tokenizer.encode(
                sentence, add_special_tokens=True)
            self.max_len = max(self.max_len, len(input_ids))
        return self.max_len
