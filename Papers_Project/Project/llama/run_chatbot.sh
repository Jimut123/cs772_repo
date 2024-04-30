#!/bin/bash

#  Increase this max_seq_len to any other number if there is no context
torchrun --nproc_per_node 1 zero_shot_chat.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 9999 --max_batch_size 6
