from transformers.configuration_utils import PretrainedConfig

class LMTPConfig(PretrainedConfig):

    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        n_head: int = 5,
        head_num_layers: int = 1,
        head_type: int = "lmtp",
        model_name: str = None,
        skip_token: int = 2,
        **kwargs,
    ):
        self.n_head = n_head
        self.head_num_layers = head_num_layers
        self.head_type = head_type
        self.model_name = model_name
        self.skip_token = skip_token
        super().__init__(
            **kwargs,
        )