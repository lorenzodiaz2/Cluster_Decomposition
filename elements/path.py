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