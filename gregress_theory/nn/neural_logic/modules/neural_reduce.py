
import torch
import torch.nn as nn

from jacinle.utils.enum import JacEnum
from jactorch.functional import quantize

__all__ = ['RelationReductionMethod', 'RelationReduction']


class RelationReductionMethod(JacEnum):
    MINMAX = 'minmax'
    PROD = 'prod'
    PROD_QUANTIZE = 'prod-quantize'


class RelationReduction(nn.Module):
    """
    Shape:
        - Input: :math:`(B, N, N-1, C)`
        - Output: :math:`(B, N, 2 * C)`
    """

    def __init__(self, method):
        super().__init__()
        self.method = RelationReductionMethod.from_string(method)

    def forward(self, relations):
        assert relations.dim() == 4
        b, n1, n2 = relations.size()[:3]

        if self.method.value.endswith('quantize'):
            if not self.training:
                relations = (relations > 0.5).float()
            else:
                relations = quantize(relations)

        if self.method.value.startswith('prod'):
            exists = 1 - torch.prod(1 - relations, dim=2)
            forall = torch.prod(relations, dim=2)
        else:
            assert self.method.value.startswith('minmax')
            exists = torch.max(relations, dim=2)[0]
            forall = torch.min(relations, dim=2)[0]

        relations = torch.stack((exists, forall), dim=-1).view(b, n1, -1)
        return relations

    def get_output_dim(self, input_dim):
        if self.method is RelationReductionMethod.MINMAX:
            return input_dim * 2
        elif self.method in (RelationReductionMethod.PROD, RelationReductionMethod.PROD_QUANTIZE):
            return input_dim * 2

    def __repr__(self):
        return '{name}({method})'.format(name=self.__class__.__name__, **self.__dict__)
