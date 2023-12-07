import matplotlib.pyplot as plt
import numpy as np

def abline(slope, intercept, label=None):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    return plt.plot(x_vals, y_vals)

def normalize(x: np.ndarray,
              ord: int | float | str = 2,
              axis: int | tuple[int, int] = None,
              keepdims: bool = False) -> np.ndarray:
    nmlzd_x = np.divide(x, np.linalg.norm(x, ord, axis, keepdims))
    nmlzd_x = np.where(np.abs(nmlzd_x) < 1e-16, 0, nmlzd_x)
    return nmlzd_x
