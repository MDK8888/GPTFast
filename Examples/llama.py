
import torch
from transformers import AutoTokenizer
from GPTFast.Core import gpt_fast

cache_config = {
    "Native":True,
    "model_config": {
        "path_to_blocks": ["model", "layers"],
    },
    "block_config": {
        "path_to_attn": ["self_attn"],   
    }, 
    "cache_position_arg_name": "cache_position"
}

def argmax(self, probabilities):
    # Use argmax to get the token with the maximum probability
    max_prob_index = torch.argmax(probabilities, dim=-1)
    return max_prob_index.view(1, 1)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = gpt_fast(model_name, sample_function=argmax, max_length = 64, cache_config=cache_config)
model = model.to(device)

prompt = "Write me a story."
prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
result = model.generate(prefix=prompt_tokens, gamma=20)
print(result)
