import numpy as np

def gd(f, g, x0, xstar, l, epsilon, iters):
    print("running smoothed clipping")
    error = []
    x = x0
    error.append((0, np.linalg.norm(x - xstar)))
    eta = 1 / l
    i = 1
    while i < iters and error[-1][1] > epsilon:
        grad = g(x)
        x = x - eta * grad
        error.append((i, abs(x - xstar)))
        i += 1
    return error