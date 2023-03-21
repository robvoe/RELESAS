from typing import Iterable


class OneHotEmbedding:
    def __init__(self, embedding_space: Iterable[str]):
        self._embedding_space = sorted(set(embedding_space))
        self._len = len(self._embedding_space)

    def __len__(self):
        return self._len

    def __call__(self, x, **kwargs) -> list[int]:
        assert x in self._embedding_space
        idx = self._embedding_space.index(x)
        one_hot = [(1 if idx == i else 0) for i in range(self._len)]
        return one_hot
