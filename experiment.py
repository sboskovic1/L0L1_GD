import functions
import numpy as np
import matplotlib.pyplot as plt
import smooth_clipping as sc
import gradient_descent as gd

def main():
    print("starting experiment")
    data = []
    data.append(functions.exp1())
    epsilon = 10e-6
    iters = 1000
    results = runl0l1(sc.smoothed_clipping, data, epsilon, iters)
    plt.plot([x[0] for x in results[0]], [x[1] for x in results[0]], label="smoothed clipping")
    print("final error for smoothed: ", results[0][-1][1], " in ", results[0][-1][0], " iterations")
    data[0][4] = data[0][0](data[0][2])
    results = runL(gd.gd, data, epsilon, iters)
    plt.plot([x[0] for x in results[0]], [x[1] for x in results[0]], label="gradient descent")
    print("final error for gradient descent: ", results[0][-1][1], " in ", results[0][-1][0], " iterations")
    plt.legend()
    print("final error: ", results[0][-1][1])
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()

def runl0l1(method, data, epsilon, iters):
    results = []
    for i in range(len(data)):
        function = data[i][0]
        gradient = data[i][1]
        x0 = data[i][2]
        xstar = data[i][3]
        l0 = data[i][4]
        l1 = data[i][5]
        results.append(method(function, gradient, np.array(x0), np.array(xstar), l0, l1, epsilon, iters))
    return results

def runL(method, data, epsilon, iters):
    results = []
    for i in range(len(data)):
        function = data[i][0]
        gradient = data[i][1]
        x0 = data[i][2]
        xstar = data[i][3]
        l = data[i][4]
        results.append(method(function, gradient, np.array(x0), np.array(xstar), l, epsilon, iters))
    return results

if __name__ == "__main__":
    main()