import numpy as np

# For normalized GD with L0-L1 smoothness. Duh
def normalized_gd(f, g, x0, xstar, l0, l1, epsilon, iters):
    # print("running normalized gradient descent")
    error = []
    x = x0
    error.append((0, np.linalg.norm(f(x) - f(xstar)))) # function value diff.
    lambda_param = 0.1
    beta = 1e-6

    # Let's hope I nailed the formula
    eta_n = lambda_param / (2 * (l0 + l1 * lambda_param))
    i = 1
    while i < iters and error[-1][1] > epsilon:
        grad = g(x)
        grad_norm = np.linalg.norm(grad)
        
        # Check if gradient is really small.
        if grad_norm < epsilon:
            break            
        
        # Normalized gradient update this had better be right.
        h_n = eta_n / (grad_norm + beta)
        x = x - h_n * grad        
        
        error.append((i, np.linalg.norm(f(x) - f(xstar))))
        i += 1
    return error