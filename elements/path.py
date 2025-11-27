import numpy as np


class Path:
    GRID_SIDE: int | None = None

    @classmethod
    def set_grid_side(cls, grid_side: int):
        cls.GRID_SIDE = grid_side

    def __init__(self, visits):
        self.visits = visits
        n = Path.GRID_SIDE

        # (i, j) -> i * n + j
        self.encoded = np.fromiter(
            (i * n + j for (i, j) in self.visits),
            dtype=np.int32,
            count=len(self.visits)
        )

    def __str__(self):
        return self.visits.__str__()

    def compare(self, other) -> int:
        a = self.encoded
        b = other.encoded
        m = min(len(a), len(b))
        return int(np.sum(a[:m] == b[:m]))
        # return sum(1 for v in self.visits if v in other.visits)

    def get_or_default(self, t):
        return self.visits[t] if t <= len(self.visits) - 1 else None