import numpy as np


def normalize(x: np.ndarray,
              ord: int | float | str = 2,
              axis: int | tuple[int, int] = None,
              keepdims: bool = False) -> np.ndarray:
    nmlzd_x = np.divide(x, np.linalg.norm(x, ord, axis, keepdims))
    nmlzd_x = np.where(np.abs(nmlzd_x) < 1e-16, 0, nmlzd_x)
    return nmlzd_x
