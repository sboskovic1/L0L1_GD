import numpy as np

nu = .56713

def smoothed_clipping(f, g, x0, xstar, l0, l1, epsilon, iters):
    print("running smoothed clipping")
    error = []
    x = x0
    error.append((0, np.linalg.norm(x - xstar)))
    eta = nu / 2
    i = 1
    print(xstar)
    while i < iters and error[-1][1] > epsilon:
        grad = g(x)
        print("x: ", x.shape)
        print("grad: ", grad.shape)
        print(x)
        print((eta / (l0 + l1 * np.linalg.norm(grad))) * grad)
        print(x - (eta / (l0 + l1 * np.linalg.norm(grad))) * grad)
        x = x - (eta / (l0 + l1 * np.linalg.norm(grad))) * grad
        print("x after: ", x.shape)
        error.append((i, np.linalg.norm(x - xstar)))
        i += 1
    return error