
import itertools

import torch
import torch.nn as nn

from jactorch.functional import broadcast

from ._utils import exclude_mask, mask_value

__all__ = ['Expander', 'Reducer', 'Permutation']


# Capture a free variable into predicates, implemented by broadcast
class Expander(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input, n=None):
        if self.dim == 0:
            assert n is not None
        elif n is None:
            n = input.size(self.dim)
        dim = self.dim + 1
        return broadcast(input.unsqueeze(dim), dim, n)

    def get_output_dim(self, input_dim):
        return input_dim


# Reduce out a variable via quantifiers (exists/forall), implemented by max/min-pooling
class Reducer(nn.Module):
    def __init__(self, dim, exclude_self=True, exists=True):
        super().__init__()
        self.dim = dim
        self.exclude_self = exclude_self
        self.exists = exists

    def forward(self, input):
        shape = input.size()
        inp0, inp1 = input, input
        if self.exclude_self:
            mask = exclude_mask(input, cnt=self.dim, dim=-1 - self.dim)
            inp0 = mask_value(input, mask, 0.0)
            inp1 = mask_value(input, mask, 1.0)

        if self.exists:
            shape = shape[:-2] + (shape[-1] * 2, )
            exists = torch.max(inp0, dim=-2)[0]
            forall = torch.min(inp1, dim=-2)[0]
            return torch.stack((exists, forall), dim=-1).view(shape)

        shape = shape[:-2] + (shape[-1], )
        return torch.max(inp0, dim=-2)[0].view(shape)

    def get_output_dim(self, input_dim):
        if self.exists:
            return input_dim * 2
        return input_dim


class Permutation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        if self.dim <= 1:
            return input
        nr_dims = len(input.size())
        # Assume the last dim is channel.
        index = tuple(range(nr_dims - 1))
        start_dim = nr_dims - 1 - self.dim
        assert start_dim > 0
        res = []
        for i in itertools.permutations(index[start_dim:]):
            p = index[:start_dim] + i + (nr_dims - 1,)
            res.append(input.permute(p))
        return torch.cat(res, dim=-1)

    def get_output_dim(self, input_dim):
        mul = 1
        for i in range(self.dim):
            mul *= i + 1
        return input_dim * mul
