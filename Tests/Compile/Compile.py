import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from GPTFast.Core.Compile.Compile import torch_compile_model
from GPTFast.Helpers import timed

torch._dynamo.reset()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.cuda()

input = "Tell me a story"
input_tokens = tokenizer.encode(input, return_tensors="pt").cuda()

N_ITERS=10

eager_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        _, eager_time = timed(lambda: model(input_tokens))
    eager_times.append(eager_time)
    print(f"eager eval time {i}: {eager_time}")

print("~" * 10)

compiled_model = torch_compile_model(model)
compiled_model.cuda()

compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        _, compile_time = timed(lambda: compiled_model(input_tokens))
    compile_times.append(compile_time)
    print(f"compile eval time {i}: {compile_time}")
print("~" * 10)