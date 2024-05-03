import torch
from pprint import pprint
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

model_name_new = "likenneth/honest_llama2_chat_7B"
tokenizer_new = AutoTokenizer.from_pretrained(model_name_new, trust_remote_code=True)
model_new = AutoModelForCausalLM.from_pretrained(model_name_new, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

q = "I ate a cherry seed. Will a cherry tree grow in my stomach?"
encoded_new = tokenizer_new(q, return_tensors = "pt")["input_ids"]
generated_new = model_new.generate(encoded_new.cuda())[0, encoded_new.shape[-1]:]
decoded_new = tokenizer_new.decode(generated_new, skip_special_tokens=True).strip()
pprint(decoded_new)

