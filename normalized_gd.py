import numpy as np

# For normalized GD with L0-L1 smoothness. Duh
def normalized_gd(f, g, x0, xstar, l0, l1, epsilon, iters):
    print("running normalized gradient descent")
    error = []
    x = x0
    error.append((0, np.linalg.norm(x - xstar)))

    # Let's hope I nailed the formula
    eta = 1 / (2 * (l0 + l1))  # Constant step size. This is correct, right?
    i = 1
    while i < iters and error[-1][1] > epsilon:
        grad = g(x)
        # Do not divide by zero!
        if np.linalg.norm(grad) < 1e-10:
            break
        # Normalize the gradient: grad / ||grad||
        normalized_grad = grad / np.linalg.norm(grad)
        x = x - eta * normalized_grad
        error.append((i, np.linalg.norm(x - xstar)))
        i += 1
    return error