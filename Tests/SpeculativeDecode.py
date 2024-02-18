import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model
import torch
from GPTFast.Core import add_speculative_decoding
from GPTFast.Helpers import timed

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
        return input_ids[:, -length:], all_probabilities_tensor.squeeze(1)
    else:
        return all_probabilities_tensor.squeeze(1)

def generate_probability_distribution(self, input_ids, length, return_text: bool = True):
    # Encode the initial token

    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pad_token_id = 50256

    total_length = length + input_ids.shape[1]
    raw_output = self.generate(input_ids, output_scores=True, attention_mask=attention_mask, max_length=total_length, return_dict_in_generate=True, pad_token_id=pad_token_id)
    logits = torch.cat(raw_output["scores"])
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    if return_text:
        output_ids = raw_output["sequences"]
        return output_ids[:, -length:], probabilities
    else:
        return probabilities

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

initial_string = "Hello, how are you?"
input_tokens = tokenizer.encode(initial_string, return_tensors="pt")

t0 = time.time()
attention_mask = torch.ones(input_tokens.shape, dtype=torch.long)
pad_token_id = 50256
raw_output = model.generate(input_tokens, attention_mask=attention_mask, max_length=56, pad_token_id=pad_token_id)
print(f"eager time: {time.time() - t0:.2f}")

add_speculative_decoding(model, draft_model, generate_probability_distribution, argmax)

t0 = time.time()
result = model.generate(cur_tokens=input_tokens, max_tokens=50, speculate_k=5, return_text=True)
print(f"speculative decode time: {time.time() - t0:.2f}")