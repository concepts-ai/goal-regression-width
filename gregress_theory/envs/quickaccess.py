
from jaclearn.rl.env import ProxyRLEnvBase
from jaclearn.rl.space import DiscreteActionSpace
from jaclearn.rl.proxy import LimitLengthProxy

from .envs import (
    Find3Env, DirectedPathFindingEnv,
    SingleClearBlockWorldEnv,
)

__all__ = ['get_find3_env', 'get_directed_pathfinding_env', 'get_single_clear_env', 'make']


def get_find3_env(nr_blocks, *args, **kwargs):
    p = Find3Env(nr_blocks)
    p = LimitLengthProxy(p, 3)
    return p


def get_directed_pathfinding_env(nr_blocks, *args, **kwargs):
    p = DirectedPathFindingEnv(nr_blocks)
    p = LimitLengthProxy(p, nr_blocks * 2)
    return p


class _MapActionProxy(ProxyRLEnvBase):
    def __init__(self, other, mapping):
        super().__init__(other)
        self._mapping = mapping

    def map_action(self, action):
        assert action < len(self._mapping)
        return self._mapping[action]

    def _get_action_space(self):
        return DiscreteActionSpace(len(self._mapping))

    def _action(self, action):
        return self.proxy.action(self.map_action(action))


def _map_blockworld_action(p, nr_blocks, exclude_self=True):
    nr_objects = nr_blocks + 1
    mapping = [(i, j) for i in range(nr_objects) for j in range(nr_objects) if (i != j or not exclude_self)]
    p = _MapActionProxy(p, mapping)
    return p


def get_single_clear_env(nr_blocks, random_order=False, exclude_self=True):
    p = SingleClearBlockWorldEnv(nr_blocks, random_order=random_order)
    p = LimitLengthProxy(p, nr_blocks * 2)
    p = _map_blockworld_action(p, nr_blocks, exclude_self=exclude_self)
    return p


def make(task, *args, **kwargs):
    if task == 'find3':
        return get_find3_env(*args, **kwargs)
    elif task == 'single-clear':
        return get_single_clear_env(*args, **kwargs)
    elif task == 'directed-pathfinding':
        return get_directed_pathfinding_env(*args, **kwargs)
    else:
        raise ValueError('Unknown task: {}.'.format(task))
