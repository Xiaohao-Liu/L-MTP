import transformers
from safetensors.torch import save_file, load_file
import os
from ..base import AutoModel
import torch
import torch.nn as nn

from ..base.kv_cache import initialize_past_key_values
from transformers.generation.utils import GenerateDecoderOnlyOutput
from ..utils import prepare_logits_processor, reset_tree_mode
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


class Vanilla(PreTrainedModel):
    config_class = PretrainedConfig
    base_model_prefix = "model"
    _no_split_modules = ["LlamaDecoderLayer", "Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True    
    def __init__(self, config, model_name, n_head=5, head_num_layers=1):
        super().__init__(config)
        self.config = transformers.AutoConfig.from_pretrained(
            model_name
        )
        
        if hasattr(self.config, "text_config"):
            self.config = self.config.text_config
            
        self.__dict__["c_model"] = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
        )
        
        if hasattr(self.__dict__["c_model"],"language_model"):
            self.__dict__["c_model"] = self.__dict__["c_model"].language_model # gemma3 is a multimodal model, so we need to get the language model
        
        self.model = self.__dict__["c_model"].model
        self.lm_head = self.__dict__["c_model"].lm_head
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side="right"
        )
                
        self.config.n_head = 0
        self.config.head_num_layers = 0
        self.config.head_type = "vanilla"
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.skip_token = self.config.skip_token =0
        self.base_model_name_or_path = model_name
                
        self.past_hidden_states = None
        self.train_mode = False
        
        if "vicuna" in self.base_model_name_or_path:
            self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
    
    def update_from_pretrained(self, model_name):
        print("[Vinalla Mode] No need to update from pretrained model")
        return None

    def get_tokenizer(self):
        return self.tokenizer
    
    @torch.autocast(device_type='cuda')
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        **kwargs,
    ):
        
        if not self.train_mode:
            with torch.inference_mode():
                outputs = self.c_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    **kwargs,
                ) # a causal lm model
        else:
            outputs = self.c_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            ) # a causal lm model
        
        return outputs
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=4096,
        log=False,
        is_llama3=False,
        **kwargs,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if attention_mask is not None: 
            assert attention_mask.shape[0] == 1, "Only support batch size 1 for now!!"
            input_ids = input_ids[attention_mask == True].unsqueeze(0)

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        # padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.model.past_key_values
            past_key_values_data = self.model.past_key_values_data
            current_length_data = self.model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.model, max_length=max_length)
            self.model.past_key_values = past_key_values
            self.model.past_key_values_data = past_key_values_data
            self.model.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.config.n_head * self.skip_token - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1
        
            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        return GenerateDecoderOnlyOutput(sequences = torch.cat([input_ids, input_ids[...,-new_token:]], dim=-1))


    
    def stream_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=4096,
        log=False,
        is_llama3=False,
        no_decode=False,
        **kwargs,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if attention_mask is not None: 
            assert attention_mask.shape[0] == 1, "Only support batch size 1 for now!!"
            input_ids = input_ids[attention_mask == True].unsqueeze(0)

        
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        input_ids = input_ids.clone()

        if hasattr(self, "past_key_values"):
            past_key_values = self.model.past_key_values
            past_key_values_data = self.model.past_key_values_data
            current_length_data = self.model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.model, max_length=max_length)
            self.model.past_key_values = past_key_values
            self.model.past_key_values_data = past_key_values_data
            self.model.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.config.n_head * self.skip_token - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            accepted_tokens = torch.tensor(1)
            new_token += accepted_tokens

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                ) if not no_decode else "",
                "accepted_tokens": accepted_tokens,
            }

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        
    def get_medusa_choice(self, model_name):
        
        return None