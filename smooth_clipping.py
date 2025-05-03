import numpy as np

nu = .56713

def smoothed_clipping(f, g, x0, xstar, l0, l1, epsilon, iters):
    # print("running smoothed clipping")
    error = []
    x = x0.reshape(-1, 1)
    xstar = xstar.reshape(-1, 1)
    error.append((0, np.linalg.norm(x - xstar)))
    eta = nu / 2
    i = 1
    while i < iters and error[-1][1] > epsilon:
        grad = g(x)
        x = x - (eta / (l0 + l1 * np.linalg.norm(grad))) * grad
        error.append((i, np.linalg.norm(f(x) - f(xstar))))
        i += 1
    return error