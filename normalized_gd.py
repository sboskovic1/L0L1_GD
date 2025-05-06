import numpy as np

def normalized_gd(f, g, x0, xstar, l0, l1, epsilon, iters):
    # Initialize
    error = []
    x = x0.reshape(-1, 1) if len(x0.shape) == 1 else x0
    xstar = xstar.reshape(-1, 1) if len(xstar.shape) == 1 else xstar
    
    # Calculate initial error
    error.append((0, np.linalg.norm(f(x) - f(xstar))))
    
    # Parameter settings from the Zhang paper, if I understood it
    if l1 > 0:
        eta_n = min(1/(20*l0), 1/(10*l1))
    else:
        # For purely L-smooth functions (L1=0) (I was getting divide by zero errors.)
        eta_n = 1/(20*l0)
    
    beta = 1e-6  # Small constant to prevent division by zero
    
    i = 1
    while i < iters and error[-1][1] > epsilon:
        # Compute gradient
        grad = g(x)
        grad_norm = np.linalg.norm(grad)
        
        # Check if gradient is very small (i.e. convergence)
        if grad_norm < 1e-10:
            break
        
        # NGD update step
        h_n = eta_n / (grad_norm + beta)
        x = x - h_n * grad
        
        # Record error
        error.append((i, np.linalg.norm(f(x) - f(xstar))))
        i += 1
        
    return error


# The version below had some silly bugs, but I'm keeping it around in the event
# that I need to go back to it for some reason.


# For normalized GD with L0-L1 smoothness. Duh
# def normalized_gd(f, g, x0, xstar, l0, l1, epsilon, iters):
#     # print("running normalized gradient descent")
#     error = []
#     x = x0
#     error.append((0, np.linalg.norm(f(x) - f(xstar)))) # function value diff.
#     lambda_param = 1
#     beta = 1e-6

#     # Let's hope I nailed the formula
#     eta_n = lambda_param / (2 * (l0 + l1 * lambda_param))
#     i = 1
#     while i < iters and error[-1][1] > epsilon:
#         grad = g(x)
#         grad_norm = np.linalg.norm(grad)
        
#         # Check if gradient is really small.
#         if grad_norm < epsilon:
#             break            
        
#         # Normalized gradient update this had better be right.
#         h_n = eta_n / (grad_norm + beta)
#         x = x - h_n * grad        
        
#         error.append((i, np.linalg.norm(f(x) - f(xstar))))
#         i += 1
#     return error