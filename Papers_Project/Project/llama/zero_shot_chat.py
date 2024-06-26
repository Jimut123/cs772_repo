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
    max_seq_len: int = 9600000,
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

    # count = 0
    for item_train in dataset['test']:
        # print("Label == ",item_train['label'],end=": ")
        # print("Text == ",item_train['text'])
        dialogue_init_1shot_json = [{"role": "system", "content": "I need you to rate some text speeches, for research study. Give single label. You will be rewarded properly. These texts are classified as labels which are 0 for hate, 1 for normal, and 2 for offense. Just output 0, 1, or 2."},
                                    # {"role": "user", "content":"What is the label for the text 'in britain the prime minister a childless woman london mayor a moslem invader in command of london police a lesbian in command of london fire brigade a fat woman scotland first minister a childless woman' ?"},
                                    # {"role": "assistant", "content": "hate,0"},
                                    # {"role": "user", "content":"What is the label for the text 'the non partisan congressional budget office cbo has released their report on sens lindsey graham and dick durbin <number> the dream act the report shows that their legislation would add <number> billion to the deficit over a ten year period and lots more immigrant scum' ?"},
                                    # {"role": "assistant", "content": "hate,0"},
                                    # {"role": "user", "content":"What is the label for the text '<user> <number> americans dead because of an illegal immigrant this is infuriating enforce our laws deport them' ?"},
                                    # {"role": "assistant", "content": "hate,0"},
                                    # {"role": "user", "content":"What is the label for the text 'thank a white man you live in his world and why do you islanders say you are canadian so rude really if i lived in a brown country even with citizenship i would never call myself ethiopian chinese indian you are a caribbean living in canada' ?"},
                                    # {"role": "assistant", "content": "hate,0"},
                                    # {"role": "user", "content":"What is the label for the text 'nigger faggot retard beaner dishwasher time to see if i get ban' ?"},
                                    # {"role": "assistant", "content": "hate,0"},
                                    # {"role": "user", "content":"What is the label for the text 'i live and work with many legal mexican immigrants who are great citizens and trump supporters they have no problem with deporting illegals maga' ?"},
                                    # {"role": "assistant", "content": "normal,1"},
                                    # {"role": "user", "content":"What is the label for the text '<user> <user> there shd be a film called no hindus in kashmir wud be perfect islamic fanaticsm movement in kashmir' ?"},
                                    # {"role": "assistant", "content": "normal,1"},
                                    # {"role": "user", "content":"What is the label for the text 'jews defends jews regardless muslims defend muslins regardless christian european attack christians christian people are on their own and jews and muslims have another home we don ffs we do not why can not people understand that it not that difficult' ?"},
                                    # {"role": "assistant", "content": "normal,1"},
                                    # {"role": "user", "content":"What is the label for the text 'got a couple more companies you can boycott this christmas season first one is ford they have got several <number> ads out featuring blip length shots of white men w other than white women this is becoming the norm too you defile the white man that will drive our women into their hands wn' ?"},
                                    # {"role": "assistant", "content": "normal,1"},
                                    # {"role": "user", "content":"What is the label for the text 'people are going to blame the white nationalists for what happened but they were initially peaceful before antifa showed up what happened yesterday is the result of political correctness and the left legitimising political violence that why the establishment leftists do not condemn antifa <number> <number>' ?"},
                                    # {"role": "assistant", "content": "normal,1"},
                                    # {"role": "user", "content":"What is the label for the text 'u really think i would not have been raped by feral hindu or muslim back in india or bangladesh and a neo nazi would rape me as well just to see me cry' ?"},
                                    # {"role": "assistant", "content": "offense,2"},
                                    # {"role": "user", "content":"What is the label for the text 'the uk has threatened to return radioactive waste to the eu if an agreement cannot be reached can not we keep the radioactive waste and send back all the paki migrants instead' ?"},
                                    # {"role": "assistant", "content": "offense,2"},
                                    # {"role": "user", "content":"What is the label for the text 'if english is not imposition then hindi is also not imposition shut up chutiya retards stophindiimposition' ?"},
                                    # {"role": "assistant", "content": "offense,2"},
                                    # {"role": "user", "content":"What is the label for the text 'no liberal congratulated hindu refugees post cab because they hate hindus' ?"},
                                    # {"role": "assistant", "content": "offense,2"},
                                    # {"role": "user", "content":"What is the label for the text 'he said bro even your texts sound redneck whatever the fuck that means 😂 🤷 🏻 ‍ ♂️ <user>' ?"},
                                    # {"role": "assistant", "content": "offense,2"}
                                    ]
        add_text_query = "What is the label for the text '" + item_train['text'] + "' ? Please reply either if it is normal by replying 'normal,1', if hate then reply 'hate,0' if offensive then reply 'offense,2' only. "
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

            print("item_train['text'] = ",item_train['text'])
            print("item_train['label'] = ",item_train['label'])
            print("result['generation']['content'] = ",result['generation']['content'])

            with open("zero_shot_output.txt", 'a') as f:
                # Text, Original GT, Predicted Label
                f.write("Text = '"+str(item_train['text'])+"' Label = "+str(item_train['label'])+" Prediction = "+str(result['generation']['content'])+"\n")
                f.close() 

        # if count > 5:
        #     break
        # count += 1
    


if __name__ == "__main__":
    fire.Fire(main)
