import torch
from transformers import AutoTokenizer
from GPTFast.Core import *
from GPTFast.Helpers import timed

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "gpt2-xl"
model = load_int8(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

initial_string = "Hello, how are you?"
input_tokens = tokenizer.encode(initial_string, return_tensors="pt").to(device)

N_ITERS=10
MAX_TOKENS=50

attention_mask = torch.ones(input_tokens.shape, dtype=torch.long).to(device)
pad_token_id = 50256

compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        _, compile_time = timed(lambda: model.generate(input_tokens, attention_mask=attention_mask, max_length=50, pad_token_id=pad_token_id))
    compile_times.append(compile_time)
    print(f"quantized eval time {i}: {compile_time}")