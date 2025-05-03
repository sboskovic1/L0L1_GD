import numpy as np

tolerance = 2

def gd(f, g, x0, xstar, l, epsilon, iters):
    # print("running GD")
    error = []
    x = x0
    error.append((0, np.linalg.norm(x - xstar)))
    eta = 1 / l
    i = 1
    diverge = 0
    diverged = False
    while i < iters and error[-1][1] > epsilon:
        grad = g(x)
        x = x - eta * grad
        if (abs(np.linalg.norm(x - xstar)) > error[-1][1]):
            diverge += 1
            if (diverge >= tolerance):
                diverged = True
                break
        else:
            diverge = 0
        error.append((i, np.linalg.norm(x - xstar)))
        i += 1
    if diverged:
        error[0] = ("diverged", i)
    return error