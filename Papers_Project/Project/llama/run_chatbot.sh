#!/bin/bash

#  Increase this max_seq_len to any other number if there is no context
# torchrun --nproc_per_node 1 llama_2_chatbot.py \
#     --ckpt_dir llama-2-7b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 9999 --max_batch_size 6

# torchrun --nproc_per_node 1 CoT_llama_2_oneshot.py \
#     --ckpt_dir llama-2-7b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 9999 --max_batch_size 6

torchrun --nproc_per_node 1 CoT_llama_2_oneshot.py \
    --ckpt_dir llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 9999 --max_batch_size 6

# torchrun --nproc_per_node 1 CoT_llama_2_fiveshot.py \
#     --ckpt_dir llama-2-7b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 9999 --max_batch_size 6


