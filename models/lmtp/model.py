import transformers
from .head import Head
from safetensors.torch import save_file, load_file
import os
from typing import Callable, Optional, Tuple, Union

from ..base import AutoModel
import torch
import torch.nn as nn

from ..base.kv_cache import initialize_past_key_values
from transformers.generation.utils import GenerateDecoderOnlyOutput, CausalLMOutputWithPast
from ..utils import prepare_logits_processor, reset_tree_mode, reset_past_key_values,  evaluate_posterior, generate_candidates
from ..medusa.utils import initialize_tree, tree_decoding, generate_tree_buffers, update_inference_inputs
from ..eagle1.utils import initialize_tree as initialize_tree_eagle, tree_decoding as tree_decoding_eagle, update_inference_inputs as update_inference_inputs_eagle

from peft import PeftModel, PeftConfig

from transformers.modeling_utils import PreTrainedModel
from .config import LMTPConfig

from transformers.loss.loss_utils import ForCausalLMLoss

class LMTPModel(PreTrainedModel):
    config_class = LMTPConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["c_model.lm_head.weight"]
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
    
    def get_input_embeddings(self):
        return self.c_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.c_model.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.c_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.c_model.lm_head = new_embeddings
    
    def __init__(self, config, model_name=None, model=None, tokenizer=None, n_head=3, head_num_layers=1, load_in_8bit=False, model_max_length=2048, train_mode=False, dtype=torch.bfloat16, decode_type="backward"):
        if model_name is not None:
            config = transformers.AutoConfig.from_pretrained(
                model_name
            )
            super().__init__(config)
            self.config = config
            
            self.__dict__["c_model"] = model
            self.model = model.model
            self.lm_head = model.lm_head
            self.generation_config = model.generation_config
            
            self.tokenizer = tokenizer
            
            self.config.model_name = model_name
            self.n_head = self.config.n_head = n_head
            self.config.head_num_layers = head_num_layers
            self.config.head_type = "lmtp"
        else:
            
            super().__init__(config)
            self.config = config
            
            base_config = transformers.AutoConfig.from_pretrained(
                config.model_name
            )
            if hasattr(base_config, "text_config"):
                base_config = base_config.text_config
            
            if model is not None:
                print("!!!!! Use provided model!")            
                model.to(dtype)
                self.__dict__["c_model"] = model
            else:
                print("!!!!! Use self-built AutoModel!")
                self.__dict__["c_model"] = AutoModel.from_pretrained(
                    config.model_name,
                    config=base_config,
                    torch_dtype=dtype,
                )
            self.model = self.c_model.model
            if not base_config.tie_word_embeddings:
                self.lm_head = self.c_model.lm_head
            else:
                self.lm_head = None
            
            self.generation_config = self.c_model.generation_config
            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    config.model_name,
                    use_fast=True,
                    model_max_length=model_max_length,
                    padding_side="right"
                )
            self.n_head = n_head

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        if hasattr(self.config,"text_config"):
            self.config.text_config.model_name = self.config.model_name
            self.config = self.config.text_config
            
        self.heads = Head(
            n_head=n_head,
            hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size,
            num_layers=head_num_layers,
            base_lm_head=self.c_model.lm_head
        ).to(dtype)
        
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.skip_token = self.config.skip_token = 3
        self.base_model_name_or_path = self.model_name = self.config.model_name
        
        self.c_model.model.past_hidden_states = None
        
        self.train_mode = train_mode
        
        self.decode_type = decode_type
        
        self.post_init()
        
        self.stage=2
        
        if "vicuna" in self.base_model_name_or_path:
            self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
        
        self.head_weight = 0.1

    @property
    def base_model(self):
        return self
    
    def update_from_adapter(self, adapter_name):
        peft_model = PeftModel.from_pretrained(self, adapter_name)
        
        self = peft_model.merge_and_unload().to(torch.bfloat16) # merge the model
        del peft_model
        torch.cuda.empty_cache()
        print("Loading model done")

    def get_tokenizer(self):
        return self.tokenizer
    
    
    def backward_decode(self, hidden_states, outputs, orig, past_key_values=None, position_ids=None):
        lmtp_logits = []
        if self.model.past_hidden_states is None:
            self.model.past_hidden_states = hidden_states[:, -2].unsqueeze(1) # get the previous one
        
        past_hidden_states = torch.stack([self.model.past_hidden_states, hidden_states[:, -1].unsqueeze(1)], dim=0) 
        
        logits_ = self.heads(past_hidden_states)
        lmtp_logits = logits_.flatten(start_dim=0, end_dim=1) # [n_head, bs, length ,hidden_size]
        self.model.past_hidden_states = past_hidden_states[-1]# update the past hidden states

        return lmtp_logits, outputs, orig
    
    def forward_decode(self, hidden_states, outputs, orig, pre_input_ids=None, past_key_values=None):
        
        with torch.inference_mode():
            # token-by-token mode
            if self.model.tree_mask is not None:
                attention_mask = self.model.tree_mask.clone()
                self.model.tree_mask = None
            else:
                attention_mask = None
            
            outputs = self.c_model(input_ids=torch.argmax(orig[:,-1:], dim=-1), attention_mask=None, past_key_values=past_key_values, position_ids=None, output_hidden_states=True)
            
            hidden_states_new = torch.stack([hidden_states[:,-1].unsqueeze(1), outputs.hidden_states[-1].clone()], dim=0)
            lmtp_logits = self.heads(hidden_states_new).flatten(start_dim=0, end_dim=1)
            lmtp_logits = torch.cat([outputs.logits[...,-1,:].unsqueeze(0).unsqueeze(-2), lmtp_logits], dim=0)
 
            self.model.current_length_data -= 1 # back to the original
            
            self.model.tree_mask = attention_mask
            
                
        return lmtp_logits, outputs, orig
    
    def loss_function(
        self,
        ori_loss: torch.Tensor,
        ori_logits: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
        **kwargs
    ) -> torch.Tensor:

        self.loss_fct = ForCausalLMLoss
        # Shift so that tokens < n predict n
        loss = 0
        
        alpha = self.head_weight if self.stage == 2 else 1
        loss_dict = {}
        
        if self.stage == 2:
            loss_lm = ori_loss 
            loss += loss_lm

        skip = self.skip_token
        for i, logits_ in enumerate(logits):
            h_logits = logits_[:, : -(skip*(i+1))].contiguous()
            h_labels = labels[..., skip*(i+1) :].contiguous()
            loss_i = self.loss_fct(logits=h_logits, labels=h_labels, vocab_size=self.config.vocab_size, **kwargs)
            loss += loss_i * alpha 

        return loss
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_orig=False,
        hidden_state_indices = None,
        direct = False,
        **kwargs,
    ):
        
        if not self.train_mode:
            with torch.inference_mode():
                outputs = self.c_model(
                    input_ids=input_ids, # [1, 4096]
                    attention_mask=attention_mask, # [1, 4096]
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    output_hidden_states=True, # if true, it will return the hidden states of the last 4 layers
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                    **kwargs,
                )
        else:
            outputs = self.c_model(
                input_ids=input_ids, # [1, 4096]
                attention_mask=attention_mask, # [1, 4096]
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=True, # if true, it will return the hidden states of the last 4 layers
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                **kwargs,
            )
        # import pdb;pdb.set_trace()
        if output_orig:
            orig = outputs.logits
        else:
            orig = None
        
        hidden_states = outputs.hidden_states[-1].clone()
        
        lmtp_logits = []
        
        if self.train_mode:
            if self.n_head == 0:
                return outputs
            for i in range(self.n_head):
                lmtp_logits.append(self.heads[i](hidden_states))
            
            lmtp_logits = torch.stack(lmtp_logits, dim=0)
            if labels is not None:
                loss = self.loss_function(ori_loss=outputs.loss, ori_logits=outputs.logits, logits=lmtp_logits, labels=labels, vocab_size=self.config.vocab_size,**kwargs)

                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=outputs.logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
                
            return lmtp_logits
        else:
            with torch.inference_mode():
                if direct:
                    return None, outputs, orig
                if output_orig:
                    if self.decode_type == "backward":
                        return self.backward_decode(hidden_states, outputs, orig)
                    elif self.decode_type == "forward":
                        return self.forward_decode(hidden_states, outputs, orig, pre_input_ids = input_ids, past_key_values=past_key_values)

                    return outputs
    
    @torch.inference_mode()
    def stream_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
        eagle_mode=False,
        no_decode=False,
        **kwargs
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if attention_mask is not None: 
            assert attention_mask.shape[0] == 1, "Only support batch size 1 for now!!"
            input_ids = input_ids[attention_mask == True].unsqueeze(0)

        if temperature > 1e-5 and eagle_mode:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        # self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self.model, "past_key_values"):
            past_key_values = self.model.past_key_values
            past_key_values_data = self.model.past_key_values_data
            current_length_data = self.model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
            past_hidden_states = self.model.past_hidden_states
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.model, max_length=max_length)
            self.model.past_key_values = past_key_values
            self.model.past_key_values_data = past_key_values_data
            self.model.current_length_data = current_length_data
            self.model.past_hidden_states = None

        input_len = input_ids.shape[1]
        
        tree_choices = self.get_medusa_choice()
        tree_buffers = generate_tree_buffers(tree_choices)
        reset_tree_mode(self)
        
        tree_logits, logits, sample_token = initialize_tree(
            # input_ids, self, past_key_values, logits_processor
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
        )

        max_length = max_length - self.n_head * self.skip_token - 10
        
        new_token = 0
        
        self.model.tree_mask = tree_buffers["tree_attn_mask"]
        
        for idx in range(max_length):
            # with Timer("all"):
            candidates, candidates_prob, tree_candidates = generate_candidates(
                tree_logits, tree_buffers["tree_indices"], tree_buffers["retrieve_indices"], sample_token, logits_processor
            )
            # with Timer("tree_decoding"):
            tree_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices"],
            ) # memory used
            best_candidate, accept_length, sample_p = evaluate_posterior(logits, candidates, logits_processor)
            # import pdb;pdb.set_trace()
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, logits, tree_logits, new_token, sample_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    tree_buffers["retrieve_indices"],
                    outputs,
                    logits,
                    tree_logits,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    logits_processor,
            )
            
            # tree logits is None, so we need to compute the tree logits here.
            
            
            # update self.model.past_hidden_states
            indices = tree_buffers["retrieve_indices"][best_candidate][accept_length]
            hidden_states = outputs.hidden_states[-1][:, indices].unsqueeze(1)
            if accept_length > 0:
                indices_past = tree_buffers["retrieve_indices"][best_candidate][accept_length-1]
                self.model.past_hidden_states = outputs.hidden_states[-1][:, indices_past].unsqueeze(1)
                            
            if self.decode_type == "backward":
                tree_logits, _, _ = self.backward_decode(hidden_states, outputs, None)
            elif self.decode_type == "forward":
                tree_logits, _, _ = self.forward_decode(hidden_states, outputs, logits, pre_input_ids = input_ids, past_key_values=past_key_values)
            
            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                ) if not no_decode else "",
                "accepted_tokens": accept_length + 1,
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
    
    @torch.inference_mode()
    def ea_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
        eagle_mode=False,
        no_decode=False,
        **kwargs
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        # self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self.model, "past_key_values"):
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
            ) = initialize_past_key_values(self.model,max_length=max_length)
            self.model.past_key_values = past_key_values
            self.model.past_key_values_data = past_key_values_data
            self.model.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree_eagle(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - len(draft_tokens) - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding_eagle(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            
            # update self.model.past_hidden_states
            indices = retrieve_indices[best_candidate][accept_length]
            hidden_states = outputs.hidden_states[-1][:, indices].unsqueeze(1)
            if accept_length > 0:
                indices_past = retrieve_indices[best_candidate][accept_length-1]
                self.model.past_hidden_states = outputs.hidden_states[-1][:, indices_past].unsqueeze(1)
                
            # import pdb;pdb.set_trace()
            
            if self.decode_type == "backward":
                tree_logits, _, _ = self.backward_decode(hidden_states, outputs, None)
                
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs_eagle(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p,
                tree_logits
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                ) if not no_decode else "",
                "accepted_tokens": accept_length + 1,
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
    
    def get_medusa_choice(self, model_name=None):
        if self.decode_type == "backward":
            return eval(self.choices[f"{self.config.head_type}{self.stage}"][self.model_name][f"{self.n_head}"]["results"]["accept_nodes"])
        else:
            return eval(self.choices[f"{self.config.head_type}{self.stage}_forward"][self.model_name][f"{self.n_head}"]["results"]["accept_nodes"])


class LMTPModel2(LMTPModel):
    
    def update_from_pretrained(self, model_name, head_name):
        print("Loading model from", model_name)
        super().update_from_pretrained(head_name)
        # self.heads.load_state_dict(load_file(os.path.join(model_name, "heads.safetensors")), strict=True)
        self.heads.to(torch.bfloat16)
        # self.c_model.lm_head.load_state_dict(load_file(os.path.join(model_name, "lm_head.safetensors")), strict=True)
        
        peft_model = PeftModel.from_pretrained(self, model_name)
        self = peft_model.merge_and_unload().to(torch.bfloat16) # merge the model
        del peft_model
        torch.cuda.empty_cache()
        print("Loading model done")
    
    def get_medusa_choice(self, model_name=None):
        return {
                "lmsys/vicuna-7b-v1.5":{
                    "backward": [(0,), (1,), (0, 0), (2,), (3,), (4,), (1, 0), (0, 1), (5,), (6,), (2, 0), (0, 2), (7,), (8,), (9,), (3, 0), (0, 3), (4, 0), (0, 4), (1, 1), (0, 0, 0), (5, 0), (0, 5), (6, 0), (0, 6), (1, 2), (2, 1), (8, 0), (7, 0), (0, 0, 1), (0, 7), (0, 8), (9, 0), (0, 9), (3, 1), (1, 3), (0, 0, 2), (4, 1), (2, 2), (1, 4), (1, 0, 0), (0, 1, 0), (0, 0, 3), (3, 2), (2, 3), (0, 0, 4), (5, 1), (1, 5), (4, 2), (2, 4), (6, 1), (1, 6), (0, 0, 5), (2, 0, 0), (0, 2, 0), (0, 0, 6), (3, 3), (8, 1), (1, 0, 1), (7, 1), (0, 1, 1), (1, 7), (1, 8)],
                    "forward": [(0,), (0, 0), (0, 1), (0, 0, 0), (0, 2), (0, 3), (0, 4), (0, 1, 0), (0, 0, 1), (0, 5), (0, 6), (0, 2, 0), (0, 0, 2), (0, 7), (0, 8), (0, 9), (0, 1, 1), (0, 3, 0), (0, 0, 3), (0, 4, 0), (0, 0, 4), (0, 0, 0, 0), (0, 5, 0), (0, 0, 5), (0, 2, 1), (0, 1, 2), (0, 6, 0), (0, 0, 6), (0, 0, 0, 1), (0, 3, 1), (0, 1, 3), (0, 7, 0), (0, 0, 7), (0, 8, 0), (0, 2, 2), (0, 0, 8), (0, 4, 1), (0, 1, 4), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 9), (0, 9, 0), (0, 0, 0, 2), (0, 5, 1), (0, 1, 5), (0, 3, 2), (0, 2, 3), (0, 0, 0, 3), (0, 6, 1), (0, 2, 4), (0, 0, 0, 4), (0, 4, 2), (0, 1, 6), (0, 2, 0, 0), (0, 1, 0, 1), (0, 0, 1, 1), (0, 0, 2, 0), (0, 7, 1), (0, 1, 7), (0, 3, 3), (0, 2, 5), (0, 8, 1), (0, 5, 2)]
                }[self.decode_type],
                "Qwen/Qwen2-7B-Instruct":{
                    "backward": [(0,), (1,), (2,), (0, 0), (3,), (4,), (5,), (1, 0), (0, 1), (6,), (7,), (8,), (2, 0), (0, 2), (9,), (3, 0), (0, 3), (1, 1), (4, 0), (0, 4), (1, 2), (2, 1), (5, 0), (0, 5), (6, 0), (0, 6), (8, 0), (7, 0), (0, 0, 0), (0, 7), (0, 8), (3, 1), (1, 3), (2, 2), (4, 1), (1, 4), (0, 0, 1), (9, 0), (0, 9), (3, 2), (2, 3), (5, 1), (1, 5), (4, 2), (2, 4), (0, 0, 2), (6, 1), (1, 6), (8, 1), (1, 0, 0), (7, 1), (1, 7), (0, 1, 0), (1, 8), (3, 3), (0, 0, 3), (5, 2), (2, 5), (3, 4), (4, 3), (0, 0, 4), (0, 0, 5), (1, 0, 1)],
                    "forward": [(0,), (0, 0), (0, 1), (0, 2), (0, 0, 0), (0, 3), (0, 4), (0, 5), (0, 1, 0), (0, 0, 1), (0, 6), (0, 7), (0, 8), (0, 2, 0), (0, 9), (0, 0, 2), (0, 1, 1), (0, 3, 0), (0, 0, 3), (0, 4, 0), (0, 0, 4), (0, 2, 1), (0, 1, 2), (0, 5, 0), (0, 0, 5), (0, 6, 0), (0, 0, 0, 0), (0, 7, 0), (0, 0, 6), (0, 3, 1), (0, 1, 3), (0, 0, 7), (0, 8, 0), (0, 0, 8), (0, 2, 2), (0, 4, 1), (0, 0, 9), (0, 9, 0), (0, 1, 4), (0, 0, 0, 1), (0, 2, 3), (0, 5, 1), (0, 3, 2), (0, 1, 5), (0, 0, 0, 2), (0, 4, 2), (0, 6, 1), (0, 2, 4), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 6), (0, 7, 1), (0, 1, 7), (0, 3, 3), (0, 8, 1), (0, 1, 8), (0, 0, 0, 3), (0, 2, 5), (0, 5, 2), (0, 4, 3), (0, 1, 9), (0, 9, 1), (0, 3, 4)]
                }[self.decode_type],
                "meta-llama/Llama-3.1-8B-Instruct":{
                    "backward":  [(0,), (1,), (2,), (0, 0), (3,), (4,), (5,), (1, 0), (0, 1), (6,), (7,), (2, 0), (0, 2), (8,), (9,), (3, 0), (0, 3), (1, 1), (4, 0), (0, 4), (5, 0), (0, 5), (2, 1), (1, 2), (6, 0), (0, 6), (0, 0, 0), (3, 1), (7, 0), (1, 3), (0, 7), (2, 2), (8, 0), (0, 8), (4, 1), (1, 4), (0, 0, 1), (9, 0), (0, 9), (3, 2), (2, 3), (1, 5), (5, 1), (0, 0, 2), (6, 1), (1, 6), (2, 4), (4, 2), (1, 0, 0), (0, 1, 0), (0, 0, 3), (3, 3), (2, 5), (5, 2), (7, 1), (1, 7), (0, 0, 4), (0, 0, 5), (8, 1), (1, 8), (6, 2), (2, 6), (3, 4)],
                    "forward": [(0,), (0, 0), (0, 1), (0, 2), (0, 0, 0), (0, 3), (0, 4), (0, 5), (0, 1, 0), (0, 0, 1), (0, 6), (0, 7), (0, 8), (0, 2, 0), (0, 0, 2), (0, 9), (0, 3, 0), (0, 0, 3), (0, 1, 1), (0, 4, 0), (0, 0, 4), (0, 5, 0), (0, 0, 5), (0, 2, 1), (0, 1, 2), (0, 6, 0), (0, 0, 6), (0, 0, 0, 0), (0, 7, 0), (0, 3, 1), (0, 1, 3), (0, 0, 7), (0, 2, 2), (0, 0, 8), (0, 8, 0), (0, 0, 0, 1), (0, 4, 1), (0, 1, 4), (0, 9, 0), (0, 0, 9), (0, 5, 1), (0, 1, 5), (0, 2, 3), (0, 3, 2), (0, 0, 0, 2), (0, 6, 1), (0, 1, 6), (0, 2, 4), (0, 4, 2), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 3), (0, 2, 5), (0, 3, 3), (0, 5, 2), (0, 0, 0, 4), (0, 7, 1), (0, 1, 7), (0, 0, 0, 5), (0, 1, 8), (0, 8, 1), (0, 2, 6), (0, 6, 2)]
                }[self.decode_type],
            }[self.model_name]