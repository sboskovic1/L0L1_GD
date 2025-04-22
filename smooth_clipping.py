import numpy as np

def smoothed_clipping(f, g, x0, xstar, l0, l1, epsilon, iters):
    print("running smoothed clipping")
    error = []
    x = x0
    error.append((0, np.linalg.norm(x - xstar)))
    eta = 0.28356 / 5
    i = 1
    while i < iters and error[-1][1] > epsilon:
        grad = g(x)
        x = x - eta / (l0 + l1 * np.linalg.norm(grad)) * grad
        error.append((i, abs(x - xstar)))
        i += 1
    return error