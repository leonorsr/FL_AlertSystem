"""Package for KD experiments that communicate distilled information instead of model weights.

This package provides a custom Flower client/strategy skeleton to support
communication modes such as `soft labels`, `hidden states` and `class prototype`.
"""

__all__ = ["custom_flwr"]
