import sys; sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from convenience import abline

def example_11_1():
    pass  # TODO

def example_11_2():
    """Example 11.2: Using the regression gradient method on a one-dimensional function."""
    def f(x): return x**2 + 1e-2*np.random.randn()
    m = 20
    delta = 1e-2
    delta_X = delta * np.random.randn(m)
    x0 = 2.0
    delta_F = np.array([f(x0 + delta_x) - f(x0) for delta_x in delta_X])
    theta = (np.linalg.pinv(delta_X[:, np.newaxis]) @ delta_F)[0]

    # Plot results
    plt.scatter(delta_X, delta_F)
    lines = abline(slope=theta, intercept=0)
    lines[0].set_label('∆f = {:.3f} x ∆x'.format(theta))
    plt.xlabel('∆x')
    plt.ylabel('∆f')
    plt.legend()
    plt.show()

def example_11_3():
    pass  # TODO
