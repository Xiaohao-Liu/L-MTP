import torch

from .lmtp.model import LMTPModel
from .medusa.model import MedusaModel
from .medusa.model_official import MedusaModel as MedusaModelOfficial
from .vanilla.model import Vanilla


def get_mtp_model(
    mtp_type: str,
    model_name_or_path: str,
    tokenizer=None,
    model=None,
    train_mode: bool = False,
    num_head: int = 5,
    head_weight: float = 0.1,
    stage1_pretrained_path: str = None,
    train_lm_head: bool = False,
):
    if hasattr(model,"language_model"):
        model = model.language_model # gemma3 is a multimodal model, so we need to get the language model
    if mtp_type == "lmtp1":
        model = LMTPModel(None, model_name=model_name_or_path, model=model, tokenizer=tokenizer, train_mode=train_mode, n_head=num_head)
        # freeze the model and lm_head
        for parameter in model.model.parameters():
            parameter.requires_grad = False
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = False
        model.heads.to(torch.bfloat16)
        model.stage=1
        model.n_head = num_head
    elif mtp_type == "lmtp2":
        model = LMTPModel.from_pretrained(stage1_pretrained_path, model=model, tokenizer=tokenizer, train_mode=train_mode)
        model.to(torch.bfloat16)  
        model.heads.heads = model.heads.heads[:num_head]
        model.n_head = num_head      
        model.train_lm_head = train_lm_head
        model.head_weight = head_weight
        print("Head weight: ", head_weight)
    return model

def get_mtp_model_inference(
    mtp_type: str,
    model_name_or_path: str,
    tokenizer=None,
    model=None,
    train_mode: bool = False,
    num_head: int = 5,
    stage1_pretrained_path: str = None,
    stage2_pretrained_path: str = None,
):
    if hasattr(model,"language_model"):
        model = model.language_model # gemma3 is a multimodal model, so we need to get the language model
    if mtp_type == "vanilla":
        model = Vanilla.from_pretrained(pretrained_model_name_or_path=model_name_or_path, model_name=model_name_or_path).to(torch.bfloat16)
    elif mtp_type == "lmtp":
        model = LMTPModel.from_pretrained(stage1_pretrained_path, model=model, tokenizer=tokenizer, train_mode=train_mode, n_head=num_head)
        model.to(torch.bfloat16)
        model.heads.heads = model.heads.heads[:num_head]
        model.n_head = num_head      
        model.stage = 1
    elif mtp_type == "lmtp_forward":
        model = LMTPModel.from_pretrained(stage1_pretrained_path, model=model, tokenizer=tokenizer, train_mode=train_mode, n_head=num_head)
        model.to(torch.bfloat16)
        model.heads.heads = model.heads.heads[:num_head]
        model.n_head = num_head      
        model.stage = 1
        model.decode_type = "forward"
    elif mtp_type == "lmtp2":
        model = LMTPModel.from_pretrained(stage1_pretrained_path, model=model, tokenizer=tokenizer, train_mode=train_mode, n_head=num_head)
        model.to(torch.bfloat16)  
        model.n_head = num_head
        model.heads.heads = model.heads.heads[:num_head]
        model.update_from_adapter(stage2_pretrained_path)
        
    return model