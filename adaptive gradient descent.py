import numpy as np
import matplotlib.pyplot as plt

def adgd(grad_f, x0, lambda0, gamma, max_iter, f=None, x_star=None, tol=1e-10):

    print("running adaptive gradient descent")
    x_k_list = []

    f_k_minus_f_star_list = []
    lambda_k_list = []
    theta_list = []

    x_prev = np.asarray(x0, dtype=float)
    lambda_prev = float(lambda0)
    theta_prev = float('inf')
    theta_list.append(theta_prev)
    theta_bool = False

    x_k_list.append(x_prev.copy())
    if f is not None and x_star is not None:
        f_prev = f(x_prev)
        f_star = f(x_star)
        f_k_minus_f_star_list.append(f_prev - f_star)
    lambda_k_list.append(lambda_prev)


    grad_x_prev = grad_f(x_prev)
    x_curr = x_prev - lambda_prev * grad_x_prev
    x_k_list.append(x_curr.copy())
    if f is not None and x_star is not None:
        f_curr = f(x_curr)
        f_k_minus_f_star_list.append(f_curr - f_star)
    lambda_k_list.append(np.nan) 

    d_squared = np.nan
    if x_star is not None:
        x1_m_xs_norm_sq = np.linalg.norm(x_curr - x_star)**2
        x1_m_x0_norm_sq = np.linalg.norm(x_curr - x_prev)**2
        if f is not None:
             f0_m_fs = f_prev - f_star
        else:
             f0_m_fs = np.nan 


    for k in range(1, max_iter):
        grad_x_curr = grad_f(x_curr)


        grad_diff_norm = np.linalg.norm(grad_x_curr - grad_x_prev)
        x_diff_norm = np.linalg.norm(x_curr - x_prev)

        term1 = np.sqrt(1 + theta_prev) * lambda_prev if theta_prev != float('inf') else float('inf')
        term2 = gamma * x_diff_norm / grad_diff_norm
        lambda_curr = min(term1, term2)

        x_next = x_curr - lambda_curr * grad_x_curr

        theta_curr = lambda_curr / lambda_prev
        theta_list.append(theta_curr)

        lambda_k_list[-1] = (lambda_curr) 

        d_squared = x1_m_xs_norm_sq + (3.0/4.0) * x1_m_x0_norm_sq + 2.0 * lambda_k_list[1] * theta_list[1] * f0_m_fs

        x_prev = x_curr
        x_curr = x_next
        grad_x_prev = grad_x_curr 
        lambda_prev = lambda_curr
        theta_prev = theta_curr

        x_k_list.append(x_curr.copy())
        if f is not None and x_star is not None:
            f_k_minus_f_star_list.append(f(x_curr) - f_star)

        lambda_k_list.append(np.nan)

    lambda_k_list.pop()

    return x_k_list, f_k_minus_f_star_list, lambda_k_list, float(d_squared) , theta_list, f, x_star
