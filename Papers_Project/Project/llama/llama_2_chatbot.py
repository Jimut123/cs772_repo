# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire
import datetime

from llama import Llama, Dialog


"""
Jimut, Tuesday, 30th April, 06:52 PM

This program tries to build a chatbot from llama-2 models, to do chain of thought reasoning.

#TODO:
1. Dump the current conversation to a file, using timestamp_name.
2. 



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



# from datasets import load_dataset, DatasetDict

# Define a function to preprocess the dataset
def preprocess(example):
    # Rename the 'post_tokens' feature to 'text'
    example['text'] = ' '.join(example['post_tokens'])
    example['label'] = max(example['annotators']['label'], key=example['annotators']['label'].count)
    del example['post_tokens']
    del example['rationales']
    return example



# create a folder if it doesnt exist
if not os.path.exists('chat_output'):
    os.makedirs('chat_output')


DIALOGUE_CONTEXT = [{"role": "system", "content": "You are now a chatbot, please behave like a chatbot, take context from previous conversation and respond accordingly."}]

# record the starting time of the conversation

now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")



def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 1,
    top_p: float = 0.9,
    # max_seq_len: int = 512,
    max_seq_len: int = 96000,
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

    text_input = "1111"
    while "0000" not in text_input:
        text_input = input("Enter the text (Enter 0000 to exit chat): ")
        if '0000' in text_input:
            break
        else:
            DIALOGUE_CONTEXT.append({"role": "user", "content": text_input})
            dialogs: List[Dialog] = [DIALOGUE_CONTEXT]

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
            # print("result['generation']['content'] = ",result['generation']['content'])

            DIALOGUE_CONTEXT.append({"role": "assistant", "content": result['generation']['content']})

            with open("chat_output/{}.txt".format(timestamp), 'a') as f:
                # Write down the input and the output of the chat
                f.write("User: "+str(text_input)+"\n")
                f.write("Bot: "+str(result['generation']['content'])+"\n")
                f.close() 
    


if __name__ == "__main__":
    fire.Fire(main)
