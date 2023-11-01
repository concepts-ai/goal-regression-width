
import numpy as np

import random as rd
import jacinle.random as random
from jaclearn.rl.env import SimpleRLEnvBase

from .block import random_generate_world
from .represent import get_coordinates, get_on_relation, get_is_ground, decorate, get_world_string

__all__ = [
    'Find3Env', 'DirectedPathFindingEnv',
    'BlockWorldEnv', 'SimpleMoveBlockWorldEnv', 'SingleClearBlockWorldEnv'
]


class Find3Env(SimpleRLEnvBase):
    def __init__(self, nr_blocks, *args, **kwargs):
        super().__init__()
        self.nr_blocks = nr_blocks
        self.is_over = False

        self._types = None
        self._match = None
        self._selected = None
        self._selected_1, self._selected_2, self._selected_3 = None, None, None
        self._groundtruth_tuple = None

    def _restart(self):
        assert self.nr_blocks > 3

        # generate a new world of objects with three types: A, B, C. Each category shuold have at least one object.

        # random three integers so that they sum up to nr_blocks
        nr_type1 = np.random.randint(1, self.nr_blocks - 1)
        nr_type2 = np.random.randint(1, self.nr_blocks - nr_type1)
        nr_type3 = self.nr_blocks - nr_type1 - nr_type2
        assert nr_type1 + nr_type2 + nr_type3 == self.nr_blocks

        self._types = np.zeros((self.nr_blocks, 3), dtype=np.float32)
        self._types[:nr_type1, 0] = 1
        self._types[nr_type1:nr_type1 + nr_type2, 1] = 1
        self._types[nr_type1 + nr_type2:, 2] = 1

        self._match = np.zeros((self.nr_blocks, self.nr_blocks), dtype=np.float32)

        type1_set = list(range(nr_type1)); rd.shuffle(type1_set)
        type2_set = list(range(nr_type1, nr_type1 + nr_type2)); rd.shuffle(type2_set)
        type3_set = list(range(nr_type1 + nr_type2, self.nr_blocks)); rd.shuffle(type3_set)

        i, j, k = type1_set.pop(), type2_set.pop(), type3_set.pop()
        self._match[i, j] = self._match[j, i] = self._match[i, k] = self._match[k, i] = self._match[j, k] = self._match[k, j] = 1
        self._groundtruth_tuple = (i, j, k)

        while True:
            link_type = np.random.randint(3)

            if (len(type1_set) > 0) + (len(type2_set) > 0) + (len(type3_set) > 0) < 2:
                break

            if link_type == 0:
                if len(type1_set) == 0 or len(type2_set) == 0:
                    continue
                i, j = type1_set.pop(), type2_set.pop()
            elif link_type == 1:
                if len(type2_set) == 0 or len(type3_set) == 0:
                    continue
                i, j = type2_set.pop(), type3_set.pop()
            else:
                if len(type3_set) == 0 or len(type1_set) == 0:
                    continue
                i, j = type3_set.pop(), type1_set.pop()
            self._match[i, j] = self._match[j, i] = 1

        self._selected = np.zeros(self.nr_blocks, dtype=np.float32)
        self._selected_1 = self._selected_2 = self._selected_3 = None

        self.is_over = False
        self._set_current_state(self._get_state())

    def _get_state(self):
        return np.concatenate([
            np.broadcast_to(self._types[:, np.newaxis], (self.nr_blocks, self.nr_blocks, 3)),
            np.broadcast_to(self._selected[:, np.newaxis, np.newaxis], (self.nr_blocks, self.nr_blocks, 1)),
            np.broadcast_to(self._match[..., np.newaxis], (self.nr_blocks, self.nr_blocks, 1)),
        ], axis=-1)

    def _action(self, action):
        assert self._selected is not None, 'You need to call restart() first.'
        if self.is_over:
            return 0, True

        x = action
        assert 0 <= x < self.nr_blocks
        obj_type = np.argmax(self._types[x])

        if obj_type == 0:
            if self._selected_1 is not None:
                self.is_over = True
                return 0, True
            self._selected_1 = x
        elif obj_type == 1:
            if self._selected_2 is not None:
                self.is_over = True
                return 0, True
            self._selected_2 = x
        else:
            if self._selected_3 is not None:
                self.is_over = True
                return 0, True
            self._selected_3 = x

        self._selected[x] = 1
        self._set_current_state(self._get_state())

        r, is_over = self._get_result()
        if is_over:
            self.is_over = True
        return r, is_over

    def _get_result(self):
        if self._selected_1 is None or self._selected_2 is None or self._selected_3 is None:
            return 0, False

        if self._match[self._selected_1, self._selected_2] == 1 and self._match[self._selected_2, self._selected_3] == 1:
            return 1, True
        else:
            return -1, True

    def get_groundtruth_action(self):
        if self._selected_1 is None:
            return self._groundtruth_tuple[0]
        elif self._selected_2 is None:
            return self._groundtruth_tuple[1]
        elif self._selected_3 is None:
            return self._groundtruth_tuple[2]

    def get_groundtruth_steps(self):
        return 3


class DirectedPathFindingEnv(SimpleRLEnvBase):
    def __init__(self, nr_blocks, *args, **kwargs):
        super().__init__()
        self.nr_blocks = nr_blocks
        self.is_over = False

        self.root_node = None
        self.target_node = None
        self.current_node = None
        self.groundtruth_path = None
        self._edges = None

    def _restart(self):
        assert self.nr_blocks > 3

        # generate a tree with nr_blocks nodes

        self.root_node = np.random.randint(self.nr_blocks)
        self._edges = np.zeros((self.nr_blocks, self.nr_blocks), dtype=np.float32)

        # generate a random tree
        nodes = list(range(self.nr_blocks))
        nodes.remove(self.root_node)
        rd.shuffle(nodes)
        added_nodes = [self.root_node]

        for node_index in nodes:
            parent_index = rd.choice(added_nodes)
            self._edges[parent_index, node_index] = 1
            added_nodes.append(node_index)

            if self._edges[parent_index].sum() >= 2:
                added_nodes.remove(parent_index)

        self.current_node = self.root_node
        self.target_node = rd.choice(added_nodes)

        def dfs(x):
            if x == self.target_node:
                return []
            for y in range(self.nr_blocks):
                if self._edges[x, y] == 1:
                    path = dfs(y)
                    if path is not None:
                        return [y] + path
            return None

        self.groundtruth_path = dfs(self.root_node)

        self.is_over = False
        self._set_current_state(self._get_state())

    def _get_state(self):
        current_node_mask = np.zeros(self.nr_blocks, dtype=np.float32)
        current_node_mask[self.current_node] = 1

        target_node_mask = np.zeros(self.nr_blocks, dtype=np.float32)
        target_node_mask[self.target_node] = 1

        return np.concatenate([
            self._edges[..., np.newaxis],
            np.broadcast_to(current_node_mask[:, np.newaxis], (self.nr_blocks, self.nr_blocks, 1)),
            np.broadcast_to(target_node_mask[:, np.newaxis], (self.nr_blocks, self.nr_blocks, 1)),
        ], axis=-1)

    def _action(self, action):
        assert self._edges is not None, 'You need to call restart() first.'
        if self.is_over:
            return 0, True

        x = action
        assert 0 <= x < self.nr_blocks

        if self._edges[self.current_node, x] == 1:
            self.current_node = x
            self._set_current_state(self._get_state())
        else:
            self.is_over = True
            return 0, True

        r, is_over = self._get_result()
        if is_over:
            self.is_over = True
        return r, is_over

    def _get_result(self):
        if self.current_node == self.target_node:
            return 1, True
        else:
            return 0, False

    def get_groundtruth_action(self):
        current_node_index = self.groundtruth_path.index(self.current_node)
        if current_node_index == len(self.groundtruth_path) - 1:
            return self.groundtruth_path[-1]
        else:
            return self.groundtruth_path[current_node_index + 1]

    def get_groundtruth_steps(self):
        return len(self.groundtruth_path)


class BlockWorldEnv(SimpleRLEnvBase):
    def __init__(self, nr_blocks, random_order=False, decorate=False, prob_unchange=0.0, prob_fall=0.0):
        """
        Args:
            nr_blocks: number of blocks.
            random_order: randomly permute the indexes of the blocks. This option prevents the models from memorizing
                the configurations.
            decorate: if True, the output coordinates with also include the world_id (default: 0) and the block index
                (starting from 0).
        """
        super().__init__()
        self.nr_blocks = nr_blocks
        self.nr_objects = nr_blocks + 1
        self.random_order = random_order
        self.decorate = decorate
        self.prob_unchange = prob_unchange
        self.prob_fall = prob_fall

    def _restart(self):
        self.world = random_generate_world(self.nr_blocks, random_order=self.random_order)
        self._set_current_state(self._get_decorated_states())
        self.is_over = False
        self.cached_result = self._get_result()

    def _get_decorated_states(self, world_id=0):
        state = get_coordinates(self.world)
        if self.decorate:
            state = decorate(state, self.nr_objects, world_id)
        return state


class SimpleMoveBlockWorldEnv(BlockWorldEnv):
    def _action(self, action):
        assert self.world is not None, 'You need to call restart() first.'
        if self.is_over:
            return 0, True
        r, is_over = self.cached_result
        if is_over:
            self.is_over = True
            return r, is_over

        x, y = action
        assert 0 <= x <= self.nr_blocks and 0 <= y <= self.nr_blocks

        p = random.rand()
        if p >= self.prob_unchange:
            if p < self.prob_unchange + self.prob_fall:
                y = self.world.blocks.inv_index(0) # fall to ground
            self.world.move(x, y)
            self._set_current_state(self._get_decorated_states())

        r, is_over = self._get_result()
        if is_over:
            self.is_over = True
        return r, is_over

    def _get_heights(self):
        coor = get_coordinates(self.world)
        height = {}
        for i in coor:
            x, y = i
            if not x in height:
                height[x] = y
            else:
                height[x] = max(height[x], y)
        heights = []
        for i in height.keys():
            heights.append(height[i])
        heights.sort()
        return heights

    def _get_result(self):
        raise NotImplementedError()


class SingleClearBlockWorldEnv(SimpleMoveBlockWorldEnv):
    def _restart(self):
        self.clear_idx = 0
        while True:
            super()._restart()

            blocks = [self.world.blocks[i] for i in range(self.nr_blocks)]
            blocks = [b for b in blocks if not b.is_ground]
            non_clear_blocks = [b for b in blocks if len(b.children) > 0]
            if len(non_clear_blocks) == 0:
                continue

            idx = random.choice_list(non_clear_blocks).index
            self.clear_idx = idx
            self.is_over = False
            self._set_current_state(self._get_decorated_states())
            self.cached_result = self._get_result()
            break

    def _get_decorated_states(self):
        on = get_on_relation(self.world)
        ground = get_is_ground(self.world)
        clear = 1 - on.max(0)
        clear_goal = np.zeros_like(ground)
        clear_goal[self.clear_idx] = 1

        return np.stack([on, np.broadcast_to(clear_goal[:, None], on.shape), np.broadcast_to(clear[:, None], on.shape), np.broadcast_to(ground[:, None], on.shape)], axis=-1)

    def _get_result(self):
        block = self.world.blocks[self.world.blocks.inv_index(self.clear_idx)]
        if len(block.children) > 0:
            return 0, False
        else:
            return 1, True

    def render(self):
        print(get_world_string(self.world))

    def get_groundtruth_steps(self):
        block = self.world.blocks[self.world.blocks.inv_index(self.clear_idx)]
        count = 0

        def dfs(block):
            nonlocal count
            if len(block.children) == 0:
                return
            for child in block.children:
                count += 1
                dfs(child)

        dfs(block)
        return count
