import json
from typing import Any
from scipy.odr import Model
from general.general_solver import General_Solver


# -----------------------------
# Helpers (robust + CSV friendly)
# -----------------------------
def _jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, set):
        return sorted(_jsonable(v) for v in x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    try:
        return _jsonable(x.item())  # numpy scalar
    except Exception:
        return str(x)


def _dump(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    return json.dumps(_jsonable(x), ensure_ascii=False)


def _safe_attr(obj: Any, name: str, default=None):
    return getattr(obj, name, default)


def _safe_objVal(solver: General_Solver):
    m: Model = _safe_attr(solver, "m", None)
    if m is None:
        return None
    if getattr(m, "SolCount", 0) and getattr(m, "SolCount") > 0:
        return float(m.ObjVal)
    return None



def _safe_objBound(solver: General_Solver):
    m: Model = _safe_attr(solver, "m", None)
    if m is None:
        return None
    try:
        return float(m.ObjBound)
    except Exception:
        return None



