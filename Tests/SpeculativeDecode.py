import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model
import torch
from GPTFast.Core import add_speculative_decoding
from GPTFast.Core import add_kv_cache
from GPTFast.Helpers import timed

def argmax(self, probabilities):
    # Use argmax to get the token with the maximum probability
    max_prob_index = torch.argmax(probabilities, dim=-1)
    return max_prob_index

# Example usage
model_name = "gpt2-xl"
draft_model_name = "gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)
draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)

model = add_kv_cache(model)
draft_model = add_kv_cache(draft_model)

initial_string = "Hello, how are you?"
input_tokens = tokenizer.encode(initial_string, return_tensors="pt")

model = add_speculative_decoding(model, draft_model, argmax)

N_ITERS=10
MAX_TOKENS=50

fast_compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        res, compile_time = timed(lambda: model.generate(cur_tokens=input_tokens, max_tokens=MAX_TOKENS, speculate_k=6))
    fast_compile_times.append(compile_time)
    print(f"speculative decode eval time {i}: {compile_time}")
print("~" * 10)
