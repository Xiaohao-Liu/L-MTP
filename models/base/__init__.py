from .modeling_llama_kv import LlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM

from .modeling_mistral_kv import MistralForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as Qwen2ForCausalLM
from .modeling_gemma3_kv import Gemma3ForCausalLM, Gemma3ForConditionalGeneration

class AutoModel():
    @classmethod
    def from_pretrained(cls, model_name, config, *args, **kwargs):
        if "llama" in model_name:
            return LlamaForCausalLM.from_pretrained(model_name, config=config, *args, **kwargs)
        elif "vicuna" in model_name:
            return LlamaForCausalLM.from_pretrained(model_name, config=config, *args, **kwargs)
        elif "mixtral" in model_name:
            return MixtralForCausalLM.from_pretrained(model_name, config=config, *args, **kwargs)
        elif "zephyr" in model_name:
            return MistralForCausalLM.from_pretrained(model_name, config=config, *args, **kwargs)
        elif "qwen2" in model_name.lower():
            return Qwen2ForCausalLM.from_pretrained(model_name, config=config, *args, **kwargs)
        elif "gemma-3" in model_name.lower():
            return Gemma3ForCausalLM.from_pretrained(model_name, config=config, *args, **kwargs)
        else:
            raise ValueError(f"Model {model_name} not supported.")

        