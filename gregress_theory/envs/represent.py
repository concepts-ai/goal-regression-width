
import numpy as np

__all__ = ['get_world_string', 'get_coordinates', 'get_on_relation', 'get_is_ground', 'get_moveable', 'get_placeable', 'decorate']


def get_world_string(world):
    index_mapping = {b.index: i for i, b in enumerate(world.blocks)}
    raw_blocks = world.blocks.raw

    result = ''

    def dfs(block, indent):
        nonlocal result

        result += '{}Block #{}: (IsGround={}, Moveable={}, Placeable={})\n'.format(
            ' ' * (indent * 2), index_mapping[block.index], block.is_ground, block.moveable, block.placeable
        )
        for c in block.children:
            dfs(c, indent + 1)

    dfs(raw_blocks[0], 0)
    return result


def get_coordinates(world, absolute=False):
    coordinates = [None for _ in range(world.size)]
    raw_blocks = world.blocks.raw

    def dfs(block):
        if block.is_ground:
            coordinates[block.index] = (0, 0)
            for j, c in enumerate(block.children):
                x = world.blocks.inv_index(c.index) if absolute else j
                coordinates[c.index] = (x, 1)
                dfs(c)
        else:
            coor = coordinates[block.index]
            assert coor is not None
            x, y = coor
            for c in block.children:
                coordinates[c.index] = (x, y + 1)
                dfs(c)

    dfs(raw_blocks[0])
    coordinates = world.blocks.permute(coordinates)
    return np.array(coordinates)


def get_on_relation(world):
    on = np.zeros((world.size, world.size), dtype=np.float32)

    def dfs(block):
        if block.is_ground:
            for c in block.children:
                on[c.index, block.index] = 1
                dfs(c)
        else:
            for c in block.children:
                on[c.index, block.index] = 1
                dfs(c)
    dfs(world.blocks.raw[0])
    return on


def get_is_ground(world):
    return np.array([block.is_ground for block in world.blocks])


def get_moveable(world):
    return np.array([block.moveable for block in world.blocks])


def get_placeable(world):
    return np.array([block.placeable for block in world.blocks])


def decorate(state, nr_objects, world_id=None):
    info = []
    if world_id is not None:
        info.append(np.ones((nr_objects, 1)) * world_id)
    info.extend([np.array(range(nr_objects))[:, np.newaxis], state])
    return np.hstack(info)
