import torch
import torch.nn as nn

def torch_compile_model(model:nn.Module, spec_decode:bool = False, device:torch.device = "cpu") -> nn.Module:
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    #model.prefill = types.MethodType(torch.compile(model.prefill, dynamic=True, fullgraph=True), model)
    model = model.to(device)
    if spec_decode:
        assert hasattr(model, "draft_model"), "You have passed spec_decode = True in your torch_compile but your model doesn't have a draft model."
        draft_model = model.draft_model
        draft_model = torch.compile(draft_model, mode="reduce-overhead", fullgraph=True)
        #draft_model.prefill = types.MethodType(torch.compile(draft_model.prefill, dynamic=True, fullgraph=True), draft_model)
        draft_model = draft_model.to(device)
    return model