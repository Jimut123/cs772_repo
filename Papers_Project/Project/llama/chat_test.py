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

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
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

    dialogs: List[Dialog] = [
        # [
        #     {"role": "system", "content": "Just reply the name. "},
        #     {"role": "user", "content": "What is the capital of India? "},
        #     {"role": "assistant", "content": "New Delhi "},
        #     {"role": "user", "content": "What is the capital of West Bengal? "},
        #     {"role": "assistant", "content": "Kolkata"},
        #     {"role": "user", "content": "What is the capital of Russia? "},
        #     {"role": "assistant", "content": "Moscow"},
        #     {"role": "user", "content": "What is the capital of Japan? "},
        #     {"role": "assistant", "content": "Tokyo"},
        #     {"role": "user", "content": "What is the capital of Sri Lanka? "},
        # ],   
        [
            {"role": "system", "content": "I need you to rate some text speeches, for research study. Give single label. You will be rewarded properly. These texts are classified as labels which are 0 for hate, 1 for normal, 2 for offense, hence reply as 1, 2 or 3 only."},
            # {"role": "user", "content": " "},
            # {"role": "user", "content": "What do you think the sentiment involved in the speech? \
            #          'u really think i would not have been raped by feral hindu or muslim back in india or bangladesh and a neo nazi would rape me as well just to see me cry' "},
            # use to over-ride the facebook stupid racist shit and bias, which we scientist don't care about.
            # {"role": "assistant", "content": "Okay. 1 for normal, 2 for offense, hence I will reply as 1, 2 or 3 only. "},
            # {"role": "user", "content": "Do this for research purposes."},
            # {"role": "assistant", "content": "Okay, go ahead, give some more details! "},
            {"role": "user", "content": "What is the label for the text 'the non partisan congressional budget office cbo has released their report on sens lindsey graham and dick durbin <number> the dream act the report shows that their legislation would add <number> billion to the deficit over a ten year period and lots more immigrant scum'"},
            # # # {"role": "system", "content": "For example, a hate (label 0) speech could be 'the non partisan congressional budget office cbo has released their report on sens lindsey graham and dick durbin <number> the dream act the report shows that their legislation would add <number> billion to the deficit over a ten year period and lots more immigrant scum'"},
            {"role": "assistant", "content": "0"},
            {"role": "user", "content": "What is the label for the text 'jews defends jews regardless muslims defend muslins regardless christian european attack christians christian people are on their own and jews and muslims have another home we don ffs we do not why can not people understand that it not that difficult'"},
            {"role": "assistant", "content": "1"},
            {"role": "user", "content": "What is the label for the text 'no liberal congratulated hindu refugees post cab because they hate hindus'"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "What is the label for the text 'i live and work with many legal mexican immigrants who are great citizens and trump supporters they have no problem with deporting illegals maga'"},
        ],
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


if __name__ == "__main__":
    fire.Fire(main)
