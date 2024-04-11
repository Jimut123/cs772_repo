
import torch
import numpy as np
import os

from datasets import  load_dataset

# ds = load_dataset('json' , data_files= data_root + 'dataset.json')

ds = load_dataset('hatexplain')

from datasets import load_dataset, DatasetDict

# Define a function to preprocess the dataset
def preprocess(example):
    # Rename the 'post_tokens' feature to 'text'
    example['text'] = ' '.join(example['post_tokens'])
    example['label'] = max(example['annotators']['label'], key=example['annotators']['label'].count)
    del example['post_tokens']
    del example['rationales']
    return example

# Apply the preprocessing function to each split of the dataset
dataset = DatasetDict({
    split: ds[split].map(preprocess)
    for split in ds.keys()
})

# print("Samples from the training dataset:: ",)
# count = 1
# # 0: "hate", 1: "normal", 2: "offense"
# print("Label = 0, Hate")
# for item_train in dataset['train']:
#     if item_train['label'] == 0:
#         print("Label == ",item_train['label'],end=": ")
#         print("Text == ",item_train['text'])
#         count += 1
#         if count >10:
#             break

# count = 1
# # 0: "hate", 1: "normal", 2: "offense"
# print("Label = 1, Normal")
# for item_train in dataset['train']:
#     if item_train['label'] == 1:
#         print("Label == ",item_train['label'],end=": ")
#         print("Text == ",item_train['text'])
#         count += 1
#         if count >10:
#             break

# count = 1
# # 0: "hate", 1: "normal", 2: "offense"
# print("Label = 2, Offense")
# for item_train in dataset['train']:
#     if item_train['label'] == 2:
#         print("Label == ",item_train['label'],end=": ")
#         print("Text == ",item_train['text'])
#         count += 1
#         if count >10:
#             break

print(dataset['test'].features)
