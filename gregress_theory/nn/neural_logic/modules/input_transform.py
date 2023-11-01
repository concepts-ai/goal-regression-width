
import torch
import torch.nn as nn

from jacinle.utils.enum import JacEnum

from ._utils import meshgrid, meshgrid_exclude_self

__all__ = ['InputTransformMethod', 'InputTransform']


class InputTransformMethod(JacEnum):
    CONCAT = 'concat'
    DIFF = 'diff'
    CMP = 'cmp'


class InputTransform(nn.Module):
    def __init__(self, method, exclude_self=True):
        super().__init__()
        self.method = InputTransformMethod.from_string(method)
        self.exclude_self = exclude_self

    def forward(self, input):
        assert input.dim() == 3

        x, y = meshgrid(input, dim=1)

        if self.method is InputTransformMethod.CONCAT:
            combined = torch.cat((x, y), dim=3)
        elif self.method is InputTransformMethod.DIFF:
            combined = x - y
        elif self.method is InputTransformMethod.CMP:
            combined = torch.cat([x < y, x == y, x > y], dim=3)
        else:
            raise ValueError('Unknown input transform method: {}.'.format(self.method))

        if self.exclude_self:
            combined = meshgrid_exclude_self(combined, dim=1)
        return combined.float()

    def get_output_dim(self, input_dim):
        if self.method is InputTransformMethod.CONCAT:
            return input_dim * 2
        elif self.method is InputTransformMethod.DIFF:
            return input_dim
        elif self.method is InputTransformMethod.CMP:
            return input_dim * 3
        else:
            raise ValueError('Unknown input transform method: {}.'.format(self.method))

    def __repr__(self):
        return '{name}({method}, exclude_self={exclude_self})'.format(name=self.__class__.__name__, **self.__dict__)
