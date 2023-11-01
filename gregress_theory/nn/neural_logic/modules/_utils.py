
import torch

import torch.autograd as ag
from jactorch.functional import meshgrid, meshgrid_exclude_self

__all__ = ['meshgrid', 'meshgrid_exclude_self', 'exclude_mask', 'mask_value', 'differentiable_less', 'differentiable_greater']


def exclude_mask(input, cnt=2, dim=1):
    """
    Produce exclude mask. Specifically, for cnt=2, given an array a[i, j] of n * n, it produces
    a mask with size n * n where only a[i, j] = 1 if and only if (i != j).

    The operation is performed over [dim, dim + cnt) axes.
    """
    assert cnt > 0
    if dim < 0:
        dim += input.dim()
    n = input.size(dim)
    for i in range(1, cnt):
        assert n == input.size(dim + i)

    rng = torch.arange(0, n, dtype=torch.long, device=input.device)
    q = []
    for i in range(cnt):
        p = rng
        for j in range(cnt):
            if i != j:
                p = p.unsqueeze(j)
        p = p.expand((n,) * cnt)
        q.append(p)
    mask = q[0] == q[0]
    for i in range(cnt):
        for j in range(cnt):
            if i != j:
                mask *= q[i] != q[j]
    for i in range(dim):
        mask.unsqueeze_(0)
    for j in range(input.dim() - dim - cnt):
        mask.unsqueeze_(-1)

    return mask.expand(input.size()).float()


def mask_value(input, mask, value):
    assert input.size() == mask.size()
    return input * mask + value * (1 - mask)


class DifferentiableLess(ag.Function):
    @staticmethod
    def forward(ctx, lhs, rhs, eps=1e-6):
        value = lhs + eps < rhs
        value = value.float()
        ctx.save_for_backward(value)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        correct = grad_output > 0
        direction = correct ^ ctx.saved_tensors[0].byte()
        direction = -1 + 2 * direction.float()
        return direction * grad_output, -direction * grad_output, None


def differentiable_less(lhs, rhs, eps=1e-6):
    return DifferentiableLess.apply(lhs, rhs, eps)


def differentiable_greater(lhs, rhs, eps=1e-6):
    return differentiable_less(-lhs, -rhs, eps)
