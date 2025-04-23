import functions
import numpy as np
import matplotlib.pyplot as plt
import smooth_clipping as sc
import gradient_descent as gd
import normalized_gd as ngd

def main():
    # stefan()
    cyrus()

def stefan():
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

# stefan I directly piggy backed off your work for mine. Not above admitting that.
def cyrus():
    print("starting normalized gradient experiment")
    data = []
    data.append(functions.cquad_func())
    epsilon = 10e-6
    iters = 1000
    
    # Run normalized gradient descent
    results_ngd = runl0l1(ngd.normalized_gd, data, epsilon, iters)
    plt.plot([x[0] for x in results_ngd[0]], [x[1] for x in results_ngd[0]], label="normalized gradient")
    print("final error for normalized: ", results_ngd[0][-1][1], " in ", results_ngd[0][-1][0], " iterations")
    
    # Run smoothed clipping (trying to compare)
    results_sc = runl0l1(sc.smoothed_clipping, data, epsilon, iters)
    plt.plot([x[0] for x in results_sc[0]], [x[1] for x in results_sc[0]], label="smoothed clipping")
    print("final error for smoothed: ", results_sc[0][-1][1], " in ", results_sc[0][-1][0], " iterations")
    
    # Run standard gradient descent
    data_gd = data.copy()
    data_gd[0] = data[0].copy()
    data_gd[0][4] = data_gd[0][0](data_gd[0][2])
    results_gd = runL(gd.gd, data_gd, epsilon, iters)
    plt.plot([x[0] for x in results_gd[0]], [x[1] for x in results_gd[0]], label="gradient descent")
    print("final error for gradient descent: ", results_gd[0][-1][1], " in ", results_gd[0][-1][0], " iterations")
    
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Comparison of Optimization Methods")
    plt.show()

if __name__ == "__main__":
    main()