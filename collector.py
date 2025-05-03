import runner as r
import matplotlib.pyplot as plt
import numpy as np
import generator as gen

def experiment(runs=10):
    epsilon = 10e-6
    iters = 10
    n = 5
    data = {}
    data['gd'] = []
    data['ngd'] = []
    data['sc'] = []
    div = []
    for i in range(runs):
        # next = r.run_function(gen.norm2, epsilon, iters, n)
        # next = r.run_function(gen.norm4, epsilon, iters, n)
        next = r.run_function(gen.exp, epsilon, iters, n)
        if (next['gd'][0][0][0] == "diverged"):
            div.append(next['gd'][0][0][1])
        else:
            data['gd'].append(next['gd'])
        data['ngd'].append(next['ngd'])
        data['sc'].append(next['sc'])
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
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Comparison of methods")
    plt.show()
    print("GD diverged: ", len(div), " times: ", div)
    return

def main():
    experiment(1)
    return

if __name__ == "__main__":
    main()
    

