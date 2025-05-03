import numpy as np
from scipy.optimize import minimize

def genA(n):
    mat = -5 + 15 * np.random.rand(n, n)
    return mat

def genY(n):
    vec = -5 + 10 * np.random.rand(n, 1)
    return vec

def norm2(A, y):
    params = {}
    params['f'] = lambda x: np.linalg.norm(A @ x - y)**2
    params['g'] = lambda x: 2 * A.T @ (A @ x - y)
    params['gnorm'] = lambda x: np.linalg.norm(params['g'](x))
    params['x0'] = np.zeros(y.shape[0])
    params['L'] = np.linalg.norm(2 * A.T @ A)
    params['L0'] = np.linalg.norm(2 * A.T @ A)
    params['L1'] = 0
    params['xstar'] = np.linalg.lstsq(A, y, rcond=None)[0]
    return params
    
def norm_power(n, dim=2):
    params = {}
    params['f'] = lambda x: np.linalg.norm(x)**(2 * n)
    params['g'] = lambda x: 2 * n * np.linalg.norm(x)**(2 * n - 2) * x if np.linalg.norm(x) != 0 else np.zeros_like(x)
    params['gnorm'] = lambda x: np.linalg.norm(params['g'](x))
    params['x0'] = np.ones(dim)  # or use [1.0, -1.0] if specific init desired
    params['L0'] = 2 * n
    params['L1'] = max(2 * n - 1, 0)
    params['xstar'] = np.zeros(dim)
    return params

def exp_inner(a):
    params = {}
    params['f'] = lambda x: np.exp(np.dot(a, x))
    params['g'] = lambda x: np.exp(np.dot(a, x)) * a
    params['gnorm'] = lambda x: np.linalg.norm(params['g'](x))
    params['x0'] = np.zeros_like(a)
    params['L0'] = 0
    params['L1'] = np.linalg.norm(a)
    # Note: Since exp(dot(a, x)) has no finite minimum without constraints, we define a surrogate "xstar"
    params['xstar'] = np.array([-25.0] * len(a))  # Or use a projection method if appropriate
    return params
    
def norm4(A, y):
    params = {}
    params['f'] = lambda x: np.linalg.norm(A @ x - y)**4
    params['g'] = lambda x: 4 * np.linalg.norm(A @ x - y)**2 * A.T @ (A @ x - y)
    params['gnorm'] = lambda x: np.linalg.norm(params['g'](x))
    params['x0'] = np.zeros(y.shape[0])
    params['L'] = max(params['gnorm'](params['x0']), params['gnorm'](params['x0'] * -1))
    params['L0'] = 4
    params['L1'] = 3
    params['xstar'] = np.linalg.lstsq(A, y, rcond=None)[0]
    return params    

def exp(A, y):
    params = {}
    params['f'] = lambda x: np.exp(y.T @ x) - y.T @ A @ x
    params['g'] = lambda x: np.exp(y.T @ x) * y - A.T @ y
    params['gnorm'] = lambda x: np.linalg.norm(params['g'](x))
    params['x0'] = np.zeros(y.shape[0])
    params['L'] = max(params['gnorm'](params['x0']), params['gnorm'](params['x0'] * -1))
    params['L0'] = 0
    params['L1'] = np.linalg.norm(y)
    params['xstar'] = minimize(params['f'], x0=np.zeros(y.shape[0])).x
    return params

# Don't know if this can even be optimized
def logexp1(A, y):
    params = {}
    params['f'] = lambda x: np.log(1 + np.exp(y.T @ x))
    return params




# def exp1():
#     params = []
#     params.append(lambda x: (math.e)**(3 * x) - x)
#     params.append(lambda x: 3 * (math.e)**(3 * x) - 1)
#     params.append(1.5)
#     params.append(-1 * math.log(3) / 3)
#     params.append(0.1)
#     params.append(3.1)
#     return params

def main():
    n = 3
    A = genA(n)
    y = genY(n)
    print("Matrix A:")
    print(A)
    print("Vector y:")
    print(y)

if __name__ == "__main__":
    main()
