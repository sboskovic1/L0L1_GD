import functions
import numpy as np
import matplotlib.pyplot as plt
import smooth_clipping as sc
import gradient_descent as gd
import normalized_gd as ngd
import functiongraphs as fg

def main():
    stefan()
    # cyrus()
    # s2()

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

def s2():
    print("starting experiment")
    data = []
    data.append(functions.exp1())
    epsilon = 10e-6
    iters = 1000
    results = runl0l1(sc.smoothed_clipping, data, epsilon, iters)
    plt.plot([x[0] for x in results[0]], [x[1] for x in results[0]], label="smoothed clipping")
    print("final error for smoothed: ", results[0][-1][1], " in ", results[0][-1][0], " iterations")
    fg.plot(functions.upperboundL0L1(functions.exp1()), 0, 150, .000001, "Upper bound for smoothed clipping")
    plt.legend()
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


def runadgd():
  y = np.array([1,-1])

  problems = {
      "norm4": functions.norm_power(5),
      "exp_inner": functions.exp_inner(np.array([5,5]))
  }

  gamma = 0.25
  max_iter = 200
  results = {}

  for name, p in problems.items():
      f, grad_f, x0, x_star = p['f'], p['g'], p['x0'], p['xstar']
      lambda0 = 0.01
      x_k, f_err, lambda_lst, D_sq, theta, f, x_star = adgd(grad_f, x0, lambda0, gamma, max_iter, f, x_star)
      grad_err = gd(f, grad_f, x0, x_star, 90, 1e-6, max_iter)
      x = np.array(x_k)
      len_graph = len(f_err)

      w = np.array([lambda_lst[k] * (1 + theta[k]) - lambda_lst[k + 1] * theta[k + 1] for k in range(1, len_graph - 2)])
      X_N_lst = []
      for i in range(1, len(w)):
          S_N = lambda_lst[1] * theta[1] + np.sum(lambda_lst[:i + 1])
          x_n_add = 1 / S_N * (lambda_lst[i] * (1 + theta[i]) + w[:i + 1] @ x[:i + 1])
          X_N_lst.append(x_n_add)

      diff_x_n_x_star_lst = [f(i) - f(x_star) for i in X_N_lst]

      L0, L1 = p['L0'], p['L1']
      D = np.sqrt(D_sq)
      nu = 0.567
      m = 1 + math.log(math.ceil(1 + L1 * D * math.e**(2 * L1 * D) / 2)) / math.log(math.sqrt(2))
      K = 2 * L1**2 * D_sq / (nu**2)
      bound_25_num = L0 * (2 + L1 * math.e**(L1 * D)) * math.e**(math.sqrt(2) * L1 * D) * D_sq
      bound_26_num = 2 * L0 * D_sq
      bound_25_lst = [bound_25_num / i for i in range(1, len_graph)]
      bound_26_lst = [
          bound_26_num / (nu * (i - m * K) - math.sqrt(2 * i) * (m + 1) * L1 * D)
          for i in range(1, len_graph)
      ]
      print(' bound constant ', (2 * m *K + 4*(m+1) * L1 *D/nu)**2)

      # Plot
      plt.figure(figsize=(10, 6))
      N_vals = np.arange(1, len_graph + 1)
      plt.plot(N_vals, f_err, label='AdGD: $f(x_k) - f(x^*)$')
      plt.plot(N_vals[:-1], grad_err, label='GD: $f(x_k) - f(x^*)$')
      plt.title(f'AdGD Convergence - {name}')
      plt.xlabel('Iteration N')
      plt.ylabel('$f(x_N) - f(x^*)$')
      plt.legend()
      plt.grid(True)
      plt.show()
if __name__ == "__main__":
    main()
