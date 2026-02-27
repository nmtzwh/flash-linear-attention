import torch
import torch.nn as nn
import torch.nn.functional as F

# Directly define and test the modified pure PyTorch elements
# to bypass the entire 'fla' package imports and their Triton dependencies.

import sys
from unittest.mock import MagicMock

# Basic mock for triton to allow imports without execution
class MockTriton:
    __version__ = "3.2.0"
    def jit(self, *args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    def autotune(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def heuristics(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class language:
        constexpr = type('constexpr', (), {})
        def arange(self, *args, **kwargs): pass
        def where(self, *args, **kwargs): pass
        def sum(self, *args, **kwargs): pass
        def sqrt(self, *args, **kwargs): pass
        def program_id(self, *args, **kwargs): pass
        def load(self, *args, **kwargs): pass
        def store(self, *args, **kwargs): pass
        def make_block_ptr(self, *args, **kwargs): pass

    class Config:
        def __init__(self, *args, **kwargs):
            pass

    def next_power_of_2(self, x): return x
    def cdiv(self, x, y): return x // y

mock_triton = MockTriton()
sys.modules['triton'] = mock_triton
sys.modules['triton.language'] = mock_triton.language

from einops import rearrange, repeat
from fla.layers.gated_deltanet import GatedDeltaNet

def test_gated_deltanet_correctness():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize the model on CPU
    batch_size = 2
    seq_len = 32
    hidden_size = 64
    
    # We configure a relatively small model to run efficiently on CPU
    model = GatedDeltaNet(
        hidden_size=hidden_size,
        expand_v=1.5,
        head_dim=16,
        num_heads=2,
        use_short_conv=True,
        conv_size=4,
        use_gate=True,
        mode='chunk'
    )
    model.eval()

    # Create dummy input
    x = torch.randn(batch_size, seq_len, hidden_size)

    with torch.no_grad():
        # Forward pass in 'chunk' mode
        model.mode = 'chunk'
        o_chunk, _, _ = model(x)
        
        # Forward pass in 'fused_recurrent' mode
        model.mode = 'fused_recurrent'
        o_recurrent, _, _ = model(x)

    # Check if the outputs from both modes are close
    # Since they are different algorithmic approaches to the same math,
    # they should yield numerically close results.
    max_diff = (o_chunk - o_recurrent).abs().max().item()
    print(f"Max difference between chunk and recurrent modes: {max_diff:.6f}")

    if max_diff < 1e-4:
        print("Success! Numerical correctness verified.")
    else:
        print("Warning: Modes differ significantly.")
        
    assert max_diff < 1e-4, f"Outputs differ too much: {max_diff}"

if __name__ == "__main__":
    test_gated_deltanet_correctness()