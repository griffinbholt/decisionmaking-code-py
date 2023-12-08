import numpy as np

def project_to_simplex(y: np.ndarray) -> np.ndarray:
    def find(u, sv):
        condition = u > ((sv - 1) / range(1,len(u)+1))
        for i in reversed(range(len(condition))): 
            if condition[i]:
                return i
    u = np.sort(y)[::-1]
    sv = np.cumsum(u)
    i = find(u, sv)
    theta = (sv[i] - 1) / (i + 1)
    x = np.maximum(y - theta, 0)
    return x
