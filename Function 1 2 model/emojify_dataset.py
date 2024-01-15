from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, RandomSampler, SequentialSampler


class EmojifyDataset():
    def __init__(self, input_ids, attention_masks, labels, batch_size=32, train_size_rate=0.95) -> None:
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.batch_size = batch_size
        self.dataset = TensorDataset(input_ids, attention_masks, labels)
        self.train_size = int(train_size_rate * len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [self.train_size, self.val_size])
        self.train_dataloader = None
        self.validation_dataloader = None
        self.data_loader()

    def data_loader(self):
        self.train_dataloader = DataLoader(self.train_dataset,
                                           sampler=RandomSampler(
                                               self.train_dataset),
                                           batch_size=self.batch_size)

        self.validation_dataloader = DataLoader(
            self.val_dataset,
            sampler=SequentialSampler(self.val_dataset),
            batch_size=self.batch_size
        )

    def get_data_loader(self):
        return self.train_dataloader, self.validation_dataloader
