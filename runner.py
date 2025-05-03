import generator as g
import numpy as np
import gradient_descent as gd
import smooth_clipping as sc
import normalized_gd as ngd

def run_function(loss, epsilon, iters, n):
    print("starting experiment")
    data = {}
    data['gd'] = []
    data['ngd'] = []
    data['sc'] = []
    f = loss(g.genA(n), g.genY(n))    
    data['gd'].append(gd.gd(f['f'], f['g'], f['x0'], f['xstar'], f['L'], epsilon, iters))
    data['ngd'].append(ngd.normalized_gd(f['f'], f['g'], f['x0'], f['xstar'], f['L0'], f['L1'], epsilon, iters))
    data['sc'].append(sc.smoothed_clipping(f['f'], f['g'], f['x0'], f['xstar'], f['L0'], f['L1'], epsilon, iters))
    return data
    
# def runl0l1(method, data, epsilon, iters):
#     results = []
#     for i in range(len(data)):
#         function = data[i][0]
#         gradient = data[i][1]
#         x0 = data[i][2]
#         xstar = data[i][3]
#         l0 = data[i][4]
#         l1 = data[i][5]
#         results.append(method(function, gradient, np.array(x0), np.array(xstar), l0, l1, epsilon, iters))
#     return results

# def runL(method, data, epsilon, iters):
#     results = []
#     for i in range(len(data)):
#         function = data[i][0]
#         gradient = data[i][1]
#         x0 = data[i][2]
#         xstar = data[i][3]
#         l = data[i][4]
#         results.append(method(function, gradient, np.array(x0), np.array(xstar), l, epsilon, iters))
#     return results
