import torch
import torch.nn as nn
from torch.nn import functional as F
class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.l_proj = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.l_proj.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()
    
    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.l_proj(x))

class HeadBlock(nn.Module):
    def __init__(self, num_layers, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.resblock = nn.Sequential(
                    *([ResBlock(hidden_size)] * num_layers)
                )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, x):
        return self.lm_head(self.resblock(x))
    
class Head(nn.Module):
    """
    Head module for the Medusa model.

    This module consists of a series of residual blocks and a final linear layer.
    It takes the output from the transformer and processes it through the blocks,
    followed by a linear transformation to produce the final output.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
        num_heads (int): The number of heads in the multi-head attention mechanism.
        num_layers (int): The number of residual blocks in the head.
    """

    def __init__(self, n_head, hidden_size, vocab_size, num_layers, base_lm_head=None, skip_token=2):
        super().__init__()

        self.n_head = n_head
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.skip_token = skip_token
        self.heads = nn.ModuleList([HeadBlock(num_layers, hidden_size, vocab_size) for _ in range(n_head)])

        if base_lm_head is not None:
            for i in range(n_head):
                # Initialize the weights of each head using the base model's weights
                # Use clone() first, then assign to avoid in-place modification issues
                # that could lead to CUDA launch failures
                weight_clone = base_lm_head.weight.data.clone()
                self.heads[i].lm_head.weight.data.copy_(weight_clone)
    
    def __getitem__(self, index):
        """
        Get the head at the specified index.

        Args:
            index (int): The index of the head to retrieve.

        Returns:
            nn.Module: The head at the specified index.
        """
        return self.heads[index]
    
    def forward(self, logits):
        """
        Forward pass of the Head module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after processing through the residual blocks and linear layer.
        """
        return torch.stack([
            head(logits) for head in self.heads
        ])
