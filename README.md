# GPTFast
Accelerate your Hugging Face Transformers 7.6-9x with GPTFast!

# Background
[GPTFast](https://github.com/pytorch-labs/gpt-fast) was originally a set of techniques developed by the PyTorch Team to accelerate the inference speed of Llama-2-7b. This pip package generalizes those techniques to all Hugging Face models. 

# Demo
GPTFast Inference Time|Eager Inference Time
--|--
![](https://github.com/MDK8888/GPTFast/assets/79173446/4d7ed04e-ba3d-49c7-aeca-8f2b96ac45a8)|![](https://github.com/MDK8888/GPTFast/assets/79173446/1a4f2236-d2f4-42c7-a689-553482871905)

# Roadmap 

- ⟳ 0.7.x (xx/xx/xx): Medusa, Speculative Sampling, Eagle
- ⟳ 0.6.x (xx/xx/xx): BitNet and 1-bit quantization, AWQ, QoQ, GGUF, HQQ
- ⟳ 0.5.x (xx/xx/xx): PagedAttention (vLLM) + FlashAttention integration 
- ⟳ 0.4.x (xx/xx/xx): Tensor parallelism + GPU distributed inference 
- ✅ 0.3.x (06/20/24): GPTQ int4 quantization and optimized int4 matmul kernels enabled for all HF models (**9x inference acceleration**) 
- ✅ 0.2.x (04/02/24): static key-value cache enabled for all HF models (**8.5x inference acceleration**)
- ✅ 0.1.x (02/22/24): torch.compile, int8 quantization, speculative decoding (**7x inference acceleration**)


# Getting Started

## WARNING: The below documentation is now deprecated with version 0.3.0. New docs will be up soon! ##


* Make sure that your python version >= 3.10, and you are on a cuda enabled device.
* Make a virtual environment on your machine and activate it.
  ```bash
  $python3 -m venv VENV_NAME
  source VENV_NAME/bin/activate #./VENV_NAME/scripts/activate if you are on Windows
  ```
* Call the following: ```pip install gptfast```
* Copy the following code into a python file:
  ```python
    import os
    import torch
    from transformers import AutoTokenizer
    from GPTFast.Core import gpt_fast
    from GPTFast.Helpers import timed
    
    torch._dynamo.reset()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
        return max_prob_index.view(1, 1)
    
    model_name = "gpt2-xl"
    draft_model_name = "gpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    initial_string = "Write me a short story."
    input_tokens = tokenizer.encode(initial_string, return_tensors="pt").to(device)
    
    N_ITERS=10
    MAX_TOKENS=50
    
    cache_config = {
        "model_config": {
            "path_to_blocks": ["transformer", "h"],
            "child_ref_in_parent_forward": ["transformer", "block"],
        },
        "block_config": {
            "path_to_attn": ["attn"],
            "child_ref_in_parent_forward": ["attn"], 
        },
        "attn_config": {
            "cache_update_config":{
                "kv_cache_condition":"if layer_past is not None",
                "key_name": "key",
                "value_name": "value",
            },
            "causal_mask_config": {
                "causal_mask_application": "conditional",
                "causal_mask_method": "_attn",
                "causal_mask_condition": "not self.is_cross_attention"
            }
        },
        "imports": ["import torch", 
                    "import transformers", 
                    "from transformers import *", 
                    "from torch import *", 
                    "from typing import *", 
                    "import types", 
                    "from transformers.modeling_outputs import *", 
                    "from torch import nn"]
    }
    
    gpt_fast_model = gpt_fast(model_name, sample_function=argmax, max_length=60, cache_config=cache_config, draft_model_name=draft_model_name)
    gpt_fast_model.to(device)
    
    fast_compile_times = []
    for i in range(N_ITERS):
        with torch.no_grad():
            res, compile_time = timed(lambda: gpt_fast_model.generate(cur_tokens=input_tokens, max_tokens=MAX_TOKENS, speculate_k=6))
        fast_compile_times.append(compile_time)
        print(f"gpt fast eval time {i}: {compile_time}")
    print("~" * 10)
    
    print(tokenizer.decode(res[0]))
  ```
* Run it and watch the magic 🪄!

# Documentation

At its core, this library provides a simple interface to LLM Inference acceleration techniques. All of the following functions can be imported from ```GPTFast.Core```:

* ```gpt_fast(model_name:str, sample_function:Callable[torch.Tensor, Dict[str, Any], torch.Tensor], max_length:int, cache_config:dict, draft_model_name:str) -> torch.nn.Module```
  * **Parameters**:
      * ```model_name```: This is the name of the Hugging Face model that you want to optimize.
      * ```sample_function```: This is a function which will take in a PyTorch Tensor which takes in a pytorch tensor as a first argument among other **sampling_kwargs and returns a Tensor of shape (1, 1).
      * ```max_length```: This is an int specifying up to how many tokens you will generating. It is recommended that you set this value higher than how many tokens you will actually generated.
      * ```cache_config```: This is a dictionary which will specify how the static key-value cache will be integrated into the model. More details for this dictionary follow below. 
      * ```draft_model_name```: This is an **optional** argument which is the name of the Hugging Face draft model which is needed for [speculative decoding](https://arxiv.org/abs/2211.17192). Note that the model and the draft model must both use the same tokenizer, and the draft model         must be **significantly** smaller to achieve inference acceleration. If ```draft_model_name``` is not specified, speculative decoding will not be applied to your model.
        
  * **Returns**:
      * An accelerated model with one method:
          * ```generate(self, cur_tokens:torch.Tensor, max_tokens:int, speculate_k:int, **sampling_kwargs) -> torch.Tensor```
              * **Parameters**:
                  * ```cur_tokens```: A PyTorch Tensor of size (1, seq_len).
                  * ```max_tokens```: An int representing how many tokens you want to generate.
                  * ```speculate_k```: An int specifying how far you want the draft model to speculate in speculative decoding.
                  * ```**sampling_kwargs```: Additional parameters that are necessary for sampling from the distribution. Should match the ```**sampling_kwargs``` of ```sample_function``` above.
              * **Returns**:
                  * The generated tokens to your prompt, a tensor with dimensions ```(1, max_tokens)```.
***

* ```load_int8(model_name:str) -> torch.nn.Module```
    * **Parameters**:
        * ```model_name```: This is a string specifying the model that you are using.
    * **Returns**:
        * An ```int8``` quantized version of your model.
***

* ```add_kv_cache(transformer:nn.Module, sampling_fn:Callable[torch.Tensor, Dict[str, Any], torch.Tensor], max_length:int, cache_config:dict) -> KVCacheModel```
    * **Parameters**:
        * ```transformer```: This is the Hugging Face model that you are adding a static key-value cache to.
        * ```sampling_fn```: This is the same as the ```sampling_function``` paramter for the ```gpt_fast``` function.
        * ```max_length```: This is the same as the ```max_length``` paramter for the ```gpt_fast``` function.
        * ```cache_config```: This is a dictionary which will specify how you **directly modify the source code of the forward pass of the model** so that a static
          cache can be accomadated. The full specifications for this dictionary are below:
          ```
           -model_config: this defines how your model should be modified to accomodate a static kv cache.
              -path_to_blocks (list[str]): Starting from the model itself, this defines the child attributes on a parent ```nn.Module``` attribute/object that we access
               to reach the blocks of a transformer.
              -child_ref_in_parent_forward (list[str]): starting from the original model, this is how each child module/attribute in ```path_to_blocks``` is referenced in
               the forward pass of the parent module/attribute. 
          
          -block_config: this defines how your block needs to be modified to accomodate a static kv cache.
              -path_to_attn (list[str]): Starting from the block itself, this defines the child attributes on a parent ```nn.Module``` attribute/object that we access to reach
               the attention layer itself.
              -child_ref_in_parent_forward (list[str]): starting from the block, this is how each child module/attribute in path_to_attn is referenced in the forward pass of the
               parent module/attribute.
      
          -attn_config: this defines how the attention layer needs to be modified to accomodate a static kv cache.
              -cache_update_config: this defines how the key-value cache updates will be modified now that it is static.
                  - kv_cache_condition (str): the condition under which a kv cache update is triggered in the source
                    code of the original forward pass of the attention layer, typically something like "if layer_past is not None."
                  - key_name (str): how the keys are originally referenced pre-update
                  - value_name (str): how the values are originally referenced pre-update
                  - new_key_name (Optional[str]): how the keys are referenced post_update. If this is not specified, this will simply be key_name. 
                  - new_value_name (Optional[str]): how the keys are referenced post_update. If this is not specified, this will simply be value_name. 
      
              -causal_mask_config: this defines how the causal mask is applied - this is necessary because your keys and values now have length ```max_length``` along the
               second-to-last dimension.
                  - causal_mask_module (str): the method of the attention layer where the causal mask is applied.
                  - causal_mask_application (Union["conditional", Any]): this is either the string "conditional" or some other value.
                  - if causal_mask_application is "conditional", you need to add the following additional keys:
                      - causal_mask_condition (str): the condition under which the causal_mask is applied.
                  - if it's not conditional, you need to add the following additional keys:
                      - causal_mask_line (str): the starting line we want to replace.
                      - num_lines (int): how many lines we want to replace including causal_mask_line
      
          -imports: these are the imports which are needed to compile your new functions after integrating a static kv cache.
          ```
          
    * **Returns**:
        * An instance of the ```KVCacheModel``` class which is essentially just your model but with a key-value cache attached for accelerated inference.
***

* ```add_speculative_decoding(model:nn.Module, draft_model:nn.Module) -> nn.Module```
    * **Parameters**:
        * ```model```: This is the KVCached version of your model.
        * ```draft_model```: This is the KVCached version of your draft model.
    * **Returns**:
        * An accelerated model with the ```generate``` method described above under the ```gpt_fast``` section.
  
  


