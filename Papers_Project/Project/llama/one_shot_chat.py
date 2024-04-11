# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog


"""
Jimut, Mon, 8th April, 1:06 AM

https://community.openai.com/t/what-exactly-does-a-system-msg-do/459409

the user messages are messages that the user wrote

the assistant messages those the bot wrote

the system message is a message that the developer wrote to tell the bot how to interpret the conversation. They're 
supposed to give instructions that can override the rest of the convo, but they're not always super reliable 
depending on what model you're using.
"""

########
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



def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 1,
    top_p: float = 0.9,
    # max_seq_len: int = 512,
    max_seq_len: int = 8192,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    count = 0
    for item_train in dataset['test']:
        # print("Label == ",item_train['label'],end=": ")
        # print("Text == ",item_train['text'])
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

        dialogs: List[Dialog] = [  
            dialogue_init_1shot_json
        ]
        
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")

            print("result['generation']['content'] = ",result['generation']['content'])

        if count > 5:
            break
        count += 1
    


if __name__ == "__main__":
    fire.Fire(main)
