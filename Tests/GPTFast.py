import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from GPTFast.Core import gpt_fast
from GPTFast.Helpers import timed

torch._dynamo.reset()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_probability_distribution_static(model, input_ids, length, return_text:bool = True):
    # Encode the initial token

    all_probabilities = []

    for _ in range(length):
        # Extract the logits from the output
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            # Get the tokens and their probabilities as a tensor
            token_probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Use the callback function for token sampling, passing any additional kwargs
        max_prob_index = torch.argmax(token_probabilities, dim=-1)
        next_token_id = max_prob_index

        # Append the sampled token to the input sequence
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(1)], dim=-1)

        # Append the probabilities to the list
        all_probabilities.append(token_probabilities.unsqueeze(0))

    # Stack the probabilities to create a tensor of size (length, vocab_size)
    all_probabilities_tensor = torch.cat(all_probabilities, dim=0)

    if return_text:
        return input_ids.squeeze(0)[-length:], all_probabilities_tensor.squeeze(1)
    else:
        return all_probabilities_tensor.squeeze(1)

def generate_probability_distribution(self, input_ids, length, return_text: bool = True):
    # Encode the initial token

    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    pad_token_id = 50256

    total_length = length + input_ids.shape[1]
    raw_output = self.generate(input_ids, output_scores=True, attention_mask=attention_mask, max_length=total_length, return_dict_in_generate=True, pad_token_id=pad_token_id)
    logits = torch.cat(raw_output["scores"])
    probabilities = torch.nn.functional.softmax(logits, dim=-1).to(device)
    if return_text:
        output_ids = raw_output["sequences"]
        return output_ids[:, -length:].to(device), probabilities
    else:
        return probabilities

def argmax_variation(self, probabilities:torch.Tensor, temperature:float = 1, k:int = 5):
    # Apply temperature scaling
    device = probabilities.device
    scaled_probabilities = probabilities / temperature

    # Ensure k is within a valid range
    k = min(k, probabilities.size(-1))

    # Get the indices of the top-k scaled probabilities along the specified dimension
    top_k_indices = torch.topk(scaled_probabilities, k, dim=-1).indices

    # Generate random indices for sampling
    random_indices = torch.randint(0, k, (1,) * probabilities.dim()).to(device)

    # Use gathered indices to get the final sampled token
    sampled_token = top_k_indices.gather(-1, random_indices).to(device)

    return sampled_token.unsqueeze(0)

def argmax(self, probabilities):
    # Use argmax to get the token with the maximum probability
    max_prob_index = torch.argmax(probabilities, dim=-1)
    return max_prob_index.unsqueeze(0)

model_name = "facebook/opt-1.3b"
#model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

draft_model_name = "facebook/opt-125m"
#draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)

initial_string = "Write me a short story."
input_tokens = tokenizer.encode(initial_string, return_tensors="pt").to(device)

N_ITERS=10
MAX_TOKENS=50

attention_mask = torch.ones(input_tokens.shape, dtype=torch.long).to(device)
pad_token_id = 50256

'''
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        _, compile_time = timed(lambda: model.generate(input_tokens, attention_mask=attention_mask, max_length=50, pad_token_id=pad_token_id))
    compile_times.append(compile_time)
    print(f"eager eval time {i}: {compile_time}")

'''
gpt_fast_model = gpt_fast(model_name, draft_model_name=draft_model_name, sample_function=argmax)
gpt_fast_model.to(device)

fast_compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        res, compile_time = timed(lambda: gpt_fast_model.generate(cur_tokens=input_tokens, max_tokens=MAX_TOKENS, speculate_k=6))
    fast_compile_times.append(compile_time)
    print(f"gpt fast eval time {i}: {compile_time}")
print("~" * 10)