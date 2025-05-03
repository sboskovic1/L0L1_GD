import generator as g
import numpy as np
import gradient_descent as gd
import smooth_clipping as sc
import normalized_gd as ngd
import adaptivegd as agd

def run_function(loss, epsilon, iters, n, safety=1):
    # print("starting experiment")
    data = {}
    data['gd'] = []
    data['ngd'] = []
    data['sc'] = []
    data['gd_safe'] = []
    data['agd'] = []
    # x0 = np.ones(n)
    f = loss(g.genA(n), g.genY(n))   
    data['gd'].append(gd.gd(f['f'], f['g'], f['x0'], f['xstar'], f['L'], epsilon, iters))
    data['gd_safe'].append(gd.gd(f['f'], f['g'], f['x0'], f['xstar'], f['L'] / safety, epsilon, iters))
    data['ngd'].append(ngd.normalized_gd(f['f'], f['g'], f['x0'], f['xstar'], f['L0'], f['L1'], epsilon, iters))
    data['sc'].append(sc.smoothed_clipping(f['f'], f['g'], f['x0'], f['xstar'], f['L0'], f['L1'], epsilon, iters))
    data['agd'].append(agd.adgd(f['g'], f['x0'], .01, .25, iters, f['f'], f['xstar'], epsilon))
    return data
    
