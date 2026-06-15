"""Gradient-reversal layer (Ganin & Lempitsky, ICML 2015).

A unary autograd function that is the identity on the forward pass and
negates the gradient (multiplied by ``lambda_``) on the backward pass.
Used to drive an adversarial domain critic on top of an encoder
representation: the critic minimises its own loss while the encoder
*maximises* it, producing a representation from which study identity
cannot be predicted.

Wrapped in an ``nn.Module`` whose ``lambda_`` attribute can be ramped
during training (DANN's standard sigmoid schedule).
"""
from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["GradientReversalFn", "GradientReversal"]


class GradientReversalFn(torch.autograd.Function):
    """Identity forward; negate-and-scale the gradient on backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversal(nn.Module):
    """Module wrapper around :class:`GradientReversalFn`.

    ``lambda_`` is a plain float attribute (not a buffer / parameter) so
    the training loop can mutate it cheaply between epochs to implement
    the DANN sigmoid schedule.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = float(lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.lambda_)
