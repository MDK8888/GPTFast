# GPTFast
Accelerate your Hugging face Transformers 6-7x with GPTFast!

# Background
[GPTFast](https://github.com/pytorch-labs/gpt-fast) was originally a set of techniques developed by the PyTorch Team to accelerate the inference speed of Llama-2-7b. This pip package generalizes those techniques to all Hugging Face models. 

# Demo
GPTFast Inference Time|Eager Inference Time
--|--
![](https://github.com/MDK8888/GPTFast/assets/79173446/4d7ed04e-ba3d-49c7-aeca-8f2b96ac45a8)|![](https://github.com/MDK8888/GPTFast/assets/79173446/1a4f2236-d2f4-42c7-a689-553482871905)

# Getting Started
* Make sure that your python version >= 3.10, and you are on a cuda enabled device.
* Make a virtual environment on your machine and activate it.
  ```bash
  $python3 -m venv VENV_NAME
  source VENV_NAME/bin/activate #./VENV_NAME/scripts/activate if you are on a Windows machine
  ```
* Call the following: ```pip install gptfast```
* Copy the following code into a python file:
  ```python
  import os
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from GPTFast.Core import gpt_fast
  from GPTFast.Helpers import timed
  
  torch._dynamo.reset()
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  
  def argmax(self, probabilities):
      # Use argmax to get the token with the maximum probability
      max_prob_index = torch.argmax(probabilities, dim=-1)
      return max_prob_index.unsqueeze(0)
  
  model_name = "gpt2-xl"
  draft_model_name = "gpt2"
  
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  initial_string = "Write me a short story."
  input_tokens = tokenizer.encode(initial_string, return_tensors="pt").to(device)
  
  N_ITERS=10
  MAX_TOKENS=50
  
  attention_mask = torch.ones(input_tokens.shape, dtype=torch.long).to(device)
  pad_token_id = 50256
  
  gpt_fast_model = gpt_fast(model_name, draft_model_name=draft_model_name, sample_function=argmax)
  gpt_fast_model.to(device)
  
  fast_compile_times = []
  for i in range(N_ITERS):
      with torch.no_grad():
          res, compile_time = timed(lambda: gpt_fast_model.generate(cur_tokens=input_tokens, max_tokens=MAX_TOKENS, speculate_k=6))
      fast_compile_times.append(compile_time)
      print(f"gpt fast eval time {i}: {compile_time}")
  print("~" * 10)
  ```
  * Run it and watch the magic 🪄!

# Documentation
At its core, this library provides a simple interface to LLM Inference acceleration techniques. All of the following functions can be imported from ```GPTFast.core```:

* ```gpt_fast(model_name:str, draft_model_name:str, sample_function:Callable) -> torch.nn.Module```
  * **Parameters**:
      * ```model_name```: This is the name of the Hugging face model that you want to optimize.
      * ```draft_model_name```: This is the name of the Hugging face draft model which is needed for [speculative decoding](https://arxiv.org/abs/2211.17192). Note that the model and the draft model must both use the same tokenizer, and the draft model must be **significantly** smaller to achieve inference acceleration.
      * ```sample function(distribution, **kwargs)```: This is a function which is used to sample from the distribution generated by the main model. This function has a mandatory parameter which is a tensor of dimension ```(seq_len, vocab_size)``` and returns a tensor of shape ```(1, 1)```. 
  * **Returns**:
      * An accelerated model with one method:
          * ```generate(self, cur_tokens:torch.Tensor, max_tokens:int, speculate_k:int, **sampling_kwargs) -> torch.Tensor```
              * **Parameters**:
                  * ```cur_tokens```: A PyTorch Tensor of size (1, seq_len).
                  * ```max_tokens```: An int representing how many tokens you want to generate.
                  * ```speculate_k```: An int specifying how far you want the draft model to speculate in speculative decoding.
                  * ```**sampling_kwargs```: Additional parameters that are necessary for sampling from the distribution. Should match the ```**kwargs``` of the ```sample``` function above.
              * **Returns**:
                  * The generated tokens to your prompt, a tensor with dimensions ```(1, max_tokens)```.

***

* ```load_int8(model_name:str) -> torch.nn.Module```
    * **Parameters**:
        * ```model_name```: This is a string specifying the model that you are using.
    * **Returns**:
        * An ```int8``` quantized version of your model.

***

* ```add_kv_cache(model_name:str) -> KVCacheModel```
    * **Parameters**:
        * ```model_name```: This is a string specifying the model that you are using.
    * **Returns**:
        * An instance of the ```KVCacheModel``` class which is essentially just your model but with a key-value cache attached for accelerated inference.

***

* ```add_speculative_decode_kv_cache(model:KVCacheModel, draft_model:KVCacheModel, sample_function:Callable) -> torch.nn.Module```
    * **Parameters**:
        * ```model```: This is the KVCached version of your model.
        * ```draft_model```: This is the KVCached version of your draft model.
        * ```sample function(distribution, **kwargs)```: same as the documentation for ```gpt_fast```.
    * **Returns**:
        * An accelerated model with the ```generate``` method described above under the ```gpt_fast``` section.
  
  


