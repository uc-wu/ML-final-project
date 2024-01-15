import json
import numpy as np
import pandas as pd
import random
import torch
import emojify_tokenize
import emojify_dataset
import emojify_model
import data_process
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, RandomSampler, SequentialSampler
from transformers import AdamW, AlbertForSequenceClassification, AlbertConfig, BertTokenizerFast, get_linear_schedule_with_warmup


def fix_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


fix_random_seed(0)

# Load data
# data_df = pd.read_csv('sentence-emoji-pair-dataset.csv', index_col=None)
data_df = pd.read_csv('dataset-new.csv', index_col=None)
train_sentences = data_df['Sentence'].tolist()
train_emojis = data_df['Emoji'].tolist()
label = data_process.convert_emoji_to_label(train_emojis)

# Tokenize Sentences and get tensor of sentences and labels.
tokenizer = BertTokenizerFast.from_pretrained(
    'bert-base-chinese')  # use pretrained tokenizer
emojify_tokenizer = emojify_tokenize.EmojifyTokenizer(
    tokenizer, train_sentences, label)
emojify_tokenizer.tokenize()
input_ids, attention_masks, labels = emojify_tokenizer.get_tensor()

# # Split data set and get data loader.
batch_size = 16
dataset = emojify_dataset.EmojifyDataset(
    input_ids, attention_masks, labels, batch_size=batch_size)
train_dataloader, validation_dataloader = dataset.get_data_loader()

# load model, False: use predtrained weights, True: use fine-tuned weights
model = emojify_model.EmojifyModel()
model.load_model(load_old_model=True)

# model.train will help us set up train_dataloader, validation_dataloader, optimizer and scheduler.
# Also we can change the epochs, learning rate and eps here.
epochs = 8
model.train(train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            epochs=epochs)

# After training, we need to save our fine-tuned model.
model.showLossFigure()
model.showAccurFigure()
model.save_model()
