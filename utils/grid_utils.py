Coord = tuple[int, int]
Quadrant = tuple[Coord, Coord]


def set_replicated_quadrants(grid_side, n_quadrants, off):
    left_up = (0, 0)
    right_down = (grid_side - 1, grid_side - 1)
    quadrants: list[Quadrant] = []
    
    if n_quadrants <= 4:
        quadrants = _divide_by_4(left_up, right_down, off)
    elif n_quadrants <= 9:
        quadrants = _divide_by_9(left_up, right_down, off)

    return quadrants



def set_divided_quadrants(grid_side, n_quadrants, off):
    left_up = (0, 0)
    right_down = (grid_side - 1, grid_side - 1)
    quadrants: list[Quadrant] = []

    if n_quadrants == 2:
        quadrants = _divide_by_2(left_up, right_down, off)
    elif n_quadrants == 3:
        quadrants = _divide_by_3(left_up, right_down, off)
    elif n_quadrants >= 4:
        q = n_quadrants // 4
        r = n_quadrants - q * 4

        quadrants = _divide_by_4(left_up, right_down, off)

        if q == 1:
            for i in range(r):
                lu, rd = quadrants[i]
                quadrants.extend(_divide_by_2(lu, rd, off))
            del quadrants[:r]

        if q == 2:
            for i in range(4):
                lu, rd = quadrants[i]
                if i < r:
                    quadrants.extend(_divide_by_3(lu, rd, off))
                else:
                    quadrants.extend(_divide_by_2(lu, rd, off))
            del quadrants[:4]

        if q == 3:
            for i in range(4):
                lu, rd = quadrants[i]
                if i < r:
                    quadrants.extend(_divide_by_4(lu, rd, off))
                else:
                    quadrants.extend(_divide_by_3(lu, rd, off))
            del quadrants[:4]

        if q == 4 and r == 0:
            for i in range(4):
                lu, rd = quadrants[i]
                quadrants.extend(_divide_by_4(lu, rd, off))
            del quadrants[:4]

    return quadrants


def _divide_by_2(
    left_up: Coord,
    right_down: Coord,
    offset: int = 0
) -> list[Quadrant]:
    top, left = left_up
    bottom, right = right_down

    left_end_col, right_start_col = _split_interval_2(left, right, offset)

    left_quadrant: Quadrant = (left_up, (bottom, left_end_col))
    right_quadrant: Quadrant = ((top, right_start_col), right_down)

    return [left_quadrant, right_quadrant]


def _divide_by_3(
    left_up: Coord,
    right_down: Coord,
    offset: int = 0
) -> list[Quadrant]:
    left_quadrant, right_quadrant = _divide_by_2(left_up, right_down, offset)
    left_up_quadrant, left_down_quadrant = _divide(left_quadrant, offset)

    return [left_up_quadrant, left_down_quadrant, right_quadrant]


def _divide_by_4(
    left_up: Coord,
    right_down: Coord,
    offset: int = 0
) -> list[Quadrant]:
    left_up_quadrant, left_down_quadrant, right_quadrant = _divide_by_3(left_up, right_down, offset)
    right_up_quadrant, right_down_quadrant = _divide(right_quadrant, offset)

    return [left_up_quadrant, left_down_quadrant, right_up_quadrant, right_down_quadrant]


def _divide(
    quadrant: Quadrant,
    offset: int = 0
) -> tuple[Quadrant, Quadrant]:
    (top, left), (bottom, right) = quadrant

    up_bottom_row, down_top_row = _split_interval_2(top, bottom, offset)

    up_quadrant: Quadrant = ((top, left), (up_bottom_row, right))
    down_quadrant: Quadrant = ((down_top_row, left), (bottom, right))

    return up_quadrant, down_quadrant



from typing import TypeAlias

Interval: TypeAlias = tuple[int, int]

def _split_interval_k(start: int, end: int, k: int, offset: int) -> list[Interval]:
    if k <= 0:
        raise ValueError("k must be > 0")
    if end < start:
        raise ValueError("empty interval")

    length = end - start + 1

    # -------- GAP (offset < 0) --------
    if offset < 0:
        gap = -offset
        usable = length - gap * (k - 1)
        if usable < k:
            raise ValueError(
                f"offset={offset} troppo negativo: non c'è spazio per {k} blocchi con gap={gap} in [{start},{end}]"
            )

        base = usable // k
        rem = usable % k
        sizes = [base + (1 if i < rem else 0) for i in range(k)]

        intervals: list[Interval] = []
        cur = start
        for sz in sizes:
            s = cur
            e = s + sz - 1
            intervals.append((s, e))
            cur = e + 1 + gap
        return intervals

    # -------- OVERLAP (offset >= 0) --------
    overlap = offset + 1  # offset=0 -> overlap 1, offset=1 -> overlap 2, ...

    base = length // k
    rem = length % k
    sizes = [base + (1 if i < rem else 0) for i in range(k)]

    intervals: list[Interval] = []
    cur = start
    for sz in sizes:
        s = cur
        e = s + sz - 1
        intervals.append((s, e))
        cur = e + 1

    # estendo a destra tutti tranne l'ultimo => overlap tra adiacenti = overlap
    for i in range(k - 1):
        s, e = intervals[i]
        intervals[i] = (s, min(end, e + overlap))

    return intervals


def _split_interval_2(start: int, end: int, offset: int) -> tuple[int, int]:
    intervals = _split_interval_k(start, end, 2, offset)
    # il type-checker non sa che sono 2: indicizziamo + assert
    assert len(intervals) == 2
    left_end = intervals[0][1]
    right_start = intervals[1][0]
    return left_end, right_start


def _split_interval_3(start: int, end: int, offset: int = 0) -> list[Interval]:
    return _split_interval_k(start, end, 3, offset)






def _divide_by_9(
    left_up: Coord,
    right_down: Coord,
    offset: int = 0
) -> list[Quadrant]:
    top, left = left_up
    bottom, right = right_down

    row_intervals = _split_interval_3(top, bottom, offset)
    col_intervals = _split_interval_3(left, right, offset)

    return [
        ((r_start, c_start), (r_end, c_end))
        for (r_start, r_end) in row_intervals
        for (c_start, c_end) in col_intervals
    ]

