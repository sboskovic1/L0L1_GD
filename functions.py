import math

def init_l0l1():
    pass

def init_L():
    pass

def init_both():
    pass

def quad1():
    params = []
    params.append(lambda x: (2 * x - 5)**2)
    params.append(8)

def exp1():
    params = []
    params.append(lambda x: (math.e)**(3 * x) - x)
    params.append(lambda x: 3 * (math.e)**(3 * x) - 1)
    params.append(1.5)
    params.append(-1 * math.log(3) / 3)
    params.append(0.1)
    params.append(3.1)
    return params

def upperboundL0L1(params):
    params = []
    params.append(lambda x: max((params[4] *(params[3] - params[2])**2) / x, params[3]**2 * (params[3] - params[2])**2))
    return params

def cquad_func():
    params = []
    params.append(lambda x: x**2)
    params.append(lambda x: 2*x)
    params.append(2.0)
    params.append(0.0)
    params.append(0.1)
    params.append(0.5)
    return params

def rosenbrock():
    params = []
    params.append(lambda x: 100*(1-x)**2 + x**2)
    params.append(lambda x: 2*x - 200*(1-x))
    params.append(0.0)
    params.append(1.0)
    params.append(2.0)
    params.append(200.0)
    return params

def plateau():
    params = []
    params.append(lambda x: x**4)
    params.append(lambda x: 4*(x**3))
    params.append(2.0)
    params.append(0.0)
    params.append(0.01)
    params.append(10.0)
    return params
