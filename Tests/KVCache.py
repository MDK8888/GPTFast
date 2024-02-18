import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from GPTFast.Core import add_kv_cache
from GPTFast.Helpers import timed

model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

model = add_kv_cache(model)

initial_string = "Hello, how are you?"
input_tokens = tokenizer.encode(initial_string, return_tensors="pt")

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def argmax(probabilities):
    # Use argmax to get the token with the maximum probability
    max_prob_index = torch.argmax(probabilities, dim=-1)
    return max_prob_index.unsqueeze(0)

N_ITERS=10
MAX_TOKENS=50

fast_compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        _, compile_time = timed(lambda: model.generate(input=input_tokens, gamma=MAX_TOKENS, sample=argmax))
    fast_compile_times.append(compile_time)
    model.rollback(0)
    print(f"KV cache model eval time {i}: {compile_time}")
print("~" * 10)



