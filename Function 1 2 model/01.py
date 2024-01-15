import numpy as np
import pandas as pd
import csv
import json


def sorting_data(data_df):

    return data_df


data_df = pd.read_csv('dataset-new.csv', index_col=None)
data_df = data_df.sort_values(by='Emoji')
data_df.to_csv("sentence-emoji-pair-dataset-new.csv", index=False)
