Coord = tuple[int, int]
Quadrant = tuple[Coord, Coord]


def set_quadrants_4_9(grid_side, n_quadrants, off):
    left_up = (0, 0)
    right_down = (grid_side - 1, grid_side - 1)
    quadrants: list[Quadrant] = []
    
    if n_quadrants <= 4:
        quadrants = _divide_by_4(left_up, right_down, off)
    elif n_quadrants <= 9:
        quadrants = _divide_by_9(left_up, right_down, off)

    return quadrants



def set_quadrants(grid_side, n_quadrants, off):
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


def _split_interval_2(
    start: int,
    end: int,
    offset: int
) -> tuple[int, int]:
    mid = (start + end) // 2

    left_end = mid + offset
    right_start = mid - offset

    left_end = max(start, min(left_end, end))
    right_start = max(start, min(right_start, end))

    return left_end, right_start


def _split_interval_3(
    start: int,
    end: int,
    offset: int = 0
) -> list[tuple[int, int]]:
    length = end - start + 1

    if offset < 0:
        gap = -offset

        usable = length - 2 * gap
        if usable % 3 != 0:
            raise ValueError(
                f"Intervallo [{start}, {end}] di lunghezza {length} "
                f"non compatibile con 3 blocchi uguali e gap={gap}"
            )

        block = usable // 3

        i1_start = start
        i1_end   = i1_start + block - 1

        i2_start = i1_end + 1 + gap
        i2_end   = i2_start + block - 1

        i3_start = i2_end + 1 + gap
        i3_end   = i3_start + block - 1

        assert i3_end == end, f"Final end {i3_end} != {end}"

        return [(i1_start, i1_end), (i2_start, i2_end), (i3_start, i3_end)]

    base = length // 3
    rem = length % 3

    sizes = [base, base, base]
    for i in range(rem):
        sizes[i] += 1

    intervals = []
    cur_start = start
    for sz in sizes:
        cur_end = cur_start + sz - 1
        intervals.append((cur_start, cur_end))
        cur_start = cur_end + 1

    return intervals



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

