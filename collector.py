import runner as r
import matplotlib.pyplot as plt
import numpy as np
import generator as gen

def experiment(func, runs=10, n=100, iters=10000):
    epsilon = 10e-6
    data = {}
    data['gd'] = []
    data['ngd'] = []
    data['sc'] = []
    data['gd_safe'] = []
    data['agd'] = []
    div = []
    div_safe = []
    safety = 100
    for i in range(runs):
        print("iteration: ", i)
        next = r.run_function(func, epsilon, iters, n, safety)
        if (next['gd'][0][0][0] == "diverged"):
            div.append(next['gd'][0][0][1])
        else:
            data['gd'].append(next['gd'])
        if (next['gd_safe'][0][0][0] == "diverged"):
            div_safe.append(next['gd_safe'][0][0][1])
        else:
            data['gd_safe'].append(next['gd_safe'])
        data['ngd'].append(next['ngd'])
        data['sc'].append(next['sc'])
        data['agd'].append(next['agd'])
    averages = {}
    for key, nested_lists in data.items():
        index_values = {}
        for group in nested_lists:
            for sublist in group:
                for idx, val in sublist:
                    if idx not in index_values:
                        index_values[idx] = []
                    index_values[idx].append(val)
        averages[key] = [sum(index_values[idx]) / len(index_values[idx]) 
                        for idx in sorted(index_values.keys())]
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(averages['gd'], label="Gradient Descent")
    plt.plot(averages['ngd'], label="Normalized Gradient Descent")
    plt.plot(averages['sc'], label="Smoothed Clipping")
    plt.plot(averages['gd_safe'], label="GD with higher L")
    plt.plot(averages['agd'], label="Adaptive GD")
    plt.legend(fontsize=8)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Comparison of methods")
    plt.show()
    print("GD diverged: ", len(div), " times: ", div)
    print("GD with higher L diverged: ", len(div_safe), " times: ", div_safe)
    return

def main():
    experiment(gen.norm2, 1, 10, 50000)
    return

if __name__ == "__main__":
    main()
    

