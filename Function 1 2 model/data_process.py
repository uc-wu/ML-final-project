import numpy as np
import pandas as pd
import csv
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def remove_invalid_data(data_df):
    jsonFile = open('cls.json', 'r')
    f = jsonFile.read()
    cls_dict = json.loads(f)
    cls = list(cls_dict.values())

    duplicated_sentence = []  # for checking if there are two same sentence
    invalid_data_idx = []
    for idx in range(len(data_df)):
        if data_df['Sentence'][idx] not in duplicated_sentence:
            duplicated_sentence.append(data_df['Sentence'][idx])
        else:
            invalid_data_idx.append(idx)

        if data_df['Emoji'][idx] not in cls and idx not in invalid_data_idx:
            invalid_data_idx.append(idx)

    data_df.drop(labels=invalid_data_idx, axis=0, inplace=True)

    return data_df


def convert_emoji_to_label(emojis=None):
    jsonFile = open('cls.json', 'r')
    f = jsonFile.read()
    cls_dict = json.loads(f)
    cls_idx_dict = {emoji: idx for idx, emoji in cls_dict.items()}

    label = []
    for emoji in emojis:
        label.append(int(cls_idx_dict[emoji]))

    return label


data_df = pd.read_csv('sentence-emoji-pair-dataset.csv', index_col=None)
data_df = remove_invalid_data(data_df)
data_df.to_csv("sentence-emoji-pair-dataset.csv", index=False)
