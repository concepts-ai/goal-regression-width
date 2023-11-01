
import jacinle.random as random

__all__ = ['Block', 'BlockWorld', 'random_generate_world']


class Block(object):
    def __init__(self, index, father=None):
        self.index = index
        self.father = father
        self.children = []

    @property
    def is_ground(self):
        return self.father is None

    @property
    def placeable(self):
        if self.is_ground:
            return True
        return len(self.children) == 0

    @property
    def moveable(self):
        if self.is_ground:
            return False
        return len(self.children) == 0

    def remove_from_father(self):
        assert self in self.father.children
        self.father.children.remove(self)
        self.father = None

    def add_to(self, other):
        self.father = other
        other.children.append(self)


class BlockStorage(object):
    def __init__(self, blocks, random_order=None):
        super().__init__()
        self._blocks = blocks
        self.set_random_order(random_order)

    def __getitem__(self, item):
        if self._random_order is None:
            return self._blocks[item]
        return self._blocks[self._random_order[item]]

    def __len__(self):
        return len(self._blocks)

    @property
    def raw(self):
        return self._blocks.copy()

    @property
    def random_order(self):
        return self._random_order

    def set_random_order(self, random_order):
        if random_order is None:
            self._random_order = None
            self._inv_random_order = None
            return

        self._random_order = random_order
        self._inv_random_order = sorted(range(len(random_order)), key=lambda x: random_order[x])

    def index(self, i):
        if self._random_order is None:
            return i
        return self._random_order[i]

    def inv_index(self, i):
        if self._random_order is None:
            return i
        return self._inv_random_order[i]

    def permute(self, array):
        if self._random_order is None:
            return array
        return [array[self._random_order[i]] for i in range(len(self._blocks))]


class BlockWorld(object):
    def __init__(self, blocks, random_order=None):
        super().__init__()
        self.blocks = BlockStorage(blocks, random_order)

    @property
    def size(self):
        return len(self.blocks)

    def move(self, x, y):
        if x != y and self.moveable(x, y):
            self.blocks[x].remove_from_father()
            self.blocks[x].add_to(self.blocks[y])

    def moveable(self, x, y):
        return self.blocks[x].moveable and self.blocks[y].placeable


# similar to random tree generation, randomly sample a valid father for new nodes
def random_generate_world(nr_blocks, random_order=False, one_stack=False):
    blocks = [Block(0, None)]
    leafs = [blocks[0]]

    for i in range(1, nr_blocks + 1):
        other = random.choice_list(leafs)
        this = Block(i)
        this.add_to(other)
        if not other.placeable or one_stack:
            leafs.remove(other)
        blocks.append(this)
        leafs.append(this)

    order = None
    if random_order:
        order = random.permutation(len(blocks))

    return BlockWorld(blocks, random_order=order)
