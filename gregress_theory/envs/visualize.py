
import sys
import functools
import numpy as np

__all__ = ['BlockWorldDisplayer']


class BlockWorldDisplayer(object):
    __display_chars__ = '0123456789' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.lower()

    def __init__(self, max_blocks=20, decorated=True, use_char=True):
        self.max_blocks = max_blocks
        self.decorated = decorated
        self.use_char = use_char

        if self.use_char:
            assert self.max_blocks <= len(self.__display_chars__)

    def display(self, coordinates, extra='', file=sys.stdout, cls=True):
        coordinates = coordinates.astype('int64')
        nr_worlds = 1
        if self.decorated:
            nr_worlds = coordinates[:, 0].max() + 1
        else:
            coordinates = np.concatenate([
                np.zeros((len(coordinates), 1), dtype=coordinates.dtype),
                np.arange(0, len(coordinates)).astype(dtype=coordinates.dtype)[:, np.newaxis],
                coordinates,
            ], axis=1)

        max_height = 0
        for i in range(nr_worlds):
            inds = np.where(np.equal(coordinates[:, 0], i))[0]
            nr_blocks = len(inds) - 1  # exclude the ground
            assert nr_blocks <= self.max_blocks
            this_height = coordinates[inds, 3].max()
            max_height = max(max_height, this_height)

        chars = [
            [' ' for _ in range(nr_worlds * (self.max_blocks + 1))]
            for _ in range(max_height + 1)
        ]

        for world_id in range(nr_worlds):
            inds = np.where(np.equal(coordinates[:, 0], world_id))[0]
            for _, i, x, y in coordinates[inds]:
                x += world_id * (self.max_blocks + 1)
                char = self.__display_chars__[i] if self.use_char else '*'
                chars[y][x] = char

        _print = functools.partial(print, file=file)
        if cls:
            _print('\x1b[2J', end='')
        _print()
        for line in reversed(chars[1:]):  # skip y=0, which is the ground
            _print(''.join(line))
        for i in range(nr_worlds):
            _print('-' * self.max_blocks, sep='', end=' ')
        _print()
        _print(extra)

    def get_display_char(self, i):
        return self.__display_chars__[i]

