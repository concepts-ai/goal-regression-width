
import torch.nn as nn

from jacinle.utils.enum import JacEnum
from jactorch.quickstart.models import MLPModel

from .input_transform import InputTransform

__all__ = ['LogicInferenceMethod', 'LogicInference', 'LogitsInference', 'TransposeLogicInference']


class LogicInferenceMethod(JacEnum):
    SKIP = 'skip'
    MLP = 'mlp'
    DNF = 'dnf'
    CNF = 'cnf'


class InferenceBase(nn.Module):
    def __init__(self, model, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.method = LogicInferenceMethod.from_string(model)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        if self.method is LogicInferenceMethod.DNF:
            raise NotImplementedError()
        elif self.method is LogicInferenceMethod.CNF:
            raise NotImplementedError()
        elif self.method is LogicInferenceMethod.MLP:
            self.layer = nn.Sequential(MLPModel(input_dim, output_dim, hidden_dim))

    def forward(self, input):
        if self.method is LogicInferenceMethod.SKIP:
            return input

        input_size = input.size()[:-1]
        input_channel = input.size(-1)

        f = input.view(-1, input_channel)
        f = self.layer(f)
        f = f.view(*input_size, -1)
        return f

    def get_output_dim(self, input_dim):
        if self.method is LogicInferenceMethod.SKIP:
            return input_dim
        return self.output_dim


class LogicInference(InferenceBase):
    def __init__(self, model, input_dim, output_dim, hidden_dim):
        super().__init__(model, input_dim, output_dim, hidden_dim)
        if self.method is LogicInferenceMethod.MLP:
            self.layer.add_module(str(len(self.layer)), nn.Sigmoid())


class LogitsInference(InferenceBase):
    pass


class TransposeLogicInference(nn.Module):
    def __init__(self, channel_dim, model, input_dim, output_dim, hidden_dim, exclude_self=True, flatten=False):
        super().__init__()
        self.channel_dim = channel_dim
        assert self.channel_dim == 1, 'Currently only support channel_dim=1.'
        self.input_transform = InputTransform('concat', exclude_self=exclude_self)
        self.logic = LogicInference(model, input_dim * 2, output_dim, hidden_dim)
        self.flatten = flatten
        assert not self.flatten, 'Currently only support flatten=False.'

    def forward(self, input):
        input = self.input_transform(input)
        return self.logic(input)

    def get_output_dim(self, input_dim):
        return self.logic.get_output_dim(input_dim)
