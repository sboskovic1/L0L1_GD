import runner as r
import matplotlib.pyplot as plt
import numpy as np
import generator as gen

def experiment(func, runs=10, n=100, iters=10000, epsilon=10e-6):
    data = {}
    data['gd'] = []
    data['ngd'] = []
    data['sc'] = []
    data['gd_safe'] = []
    data['agd'] = []
    div = {}
    div['gd'] = 0
    div['ngd'] = 0
    div['sc'] = 0
    div['agd'] = 0
    div['safe'] = 0
    safety = 100
    for i in range(runs):
        print("iteration: ", i)
        next = r.run_function(func, epsilon, iters, n, safety)
        if (next['gd'][0][-1][1] < epsilon):
            data['gd'].append(next['gd'])
        else:
            div['gd'] += 1
        if (next['gd_safe'][0][-1][1] < epsilon):
            data['gd_safe'].append(next['gd_safe'])
        else:
            div['safe'] += 1
        if (next['ngd'][0][-1][1] < epsilon):
            data['ngd'].append(next['ngd'])
        else:
            div['ngd'] += 1
        if (next['sc'][0][-1][1] < epsilon):
            data['sc'].append(next['sc'])
        else:
            div['sc'] += 1
        if (next['agd'][0][-1][1] < epsilon):
            data['agd'].append(next['agd'])
        else:
            div['agd'] += 1
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
    print(div)
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(averages['gd'], label="Gradient Descent")
    plt.plot(averages['ngd'], label="Normalized Gradient Descent")
    plt.plot(averages['sc'], label="Smoothed Clipping")
    # plt.plot(averages['gd_safe'], label="GD with higher L")
    plt.plot(averages['agd'], label="Adaptive GD")
    plt.legend(fontsize=8)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Comparison of methods")
    plt.show()
    return

def main():
    experiment(gen.exp, 20, 10, 10000, 10e-10)
    return

if __name__ == "__main__":
    main()
    

