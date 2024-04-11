#!/bin/bash

# torchrun --nproc_per_node 1 example_chat_completion.py \
#     --ckpt_dir llama-2-7b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 512 --max_batch_size 6

# torchrun --nproc_per_node 1 chat_test.py \
#     --ckpt_dir llama-2-7b/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 512 --max_batch_size 6

# torchrun --nproc_per_node 3 chat_test.py \
#     --ckpt_dir llama-2-70b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 512 --max_batch_size 6


############ Works - One shot versions

# 16387*2 VRAM of GPU
# torchrun --nproc_per_node 2 one_shot_chat.py \
#     --ckpt_dir llama-2-13b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 512 --max_batch_size 6

# 15984 VRAM of GPU
# torchrun --nproc_per_node 1 one_shot_chat.py \
#     --ckpt_dir llama-2-7b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 512 --max_batch_size 6

############ Works - 5 shot versions

# 15984 VRAM of GPU
torchrun --nproc_per_node 1 five_shot_chat.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6




