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
