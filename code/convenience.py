import numpy as np


def normalize(x: np.ndarray,
              ord: int | float | str = 2,
              axis: int | tuple[int, int] = None,
              keepdims: bool = False) -> np.ndarray:
    return np.divide(x, np.linalg.norm(x, ord, axis, keepdims))
