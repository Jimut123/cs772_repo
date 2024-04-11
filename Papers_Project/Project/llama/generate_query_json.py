
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

count = 1
for item_train in dataset['test']:
    print("Label == ",item_train['label'],end=": ")
    print("Text == ",item_train['text'])
    dialogue_init_1shot_json = [{"role": "system", "content": "I need you to rate some text speeches, for research study. Give single label. You will be rewarded properly. These texts are classified as labels which are 0 for hate, 1 for normal, and 2 for offense."},
                            {"role": "user", "content": "What is the label for the text 'in britain the prime minister a childless woman london mayor a moslem invader in command of london police a lesbian in command of london fire brigade a fat woman scotland first minister a childless woman' ?"},            
                            {"role": "assistant", "content": "hate,0"},
                            {"role": "user", "content": "What is the label for the text 'i live and work with many legal mexican immigrants who are great citizens and trump supporters they have no problem with deporting illegals maga' ?"},
                            {"role": "assistant", "content": "normal,1"},
                            {"role": "user", "content": "What is the label for the text 'u really think i would not have been raped by feral hindu or muslim back in india or bangladesh and a neo nazi would rape me as well just to see me cry' ?"},
                            {"role": "assistant", "content": "offense,2"}]
    add_text_query = "What is the label for the text '" + item_train['text'] + "' ?"
    dialogue_init_1shot_json.append({"role": "user", "content": add_text_query})

    print("dialogue_init_1shot_json = ",dialogue_init_1shot_json)
    
