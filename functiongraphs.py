import matplotlib.pyplot as plt
import functions
import new_experiments.generator as g
import numpy as np

def main():
    f = []
    f.append(g.exp(g.genA(3), g.genY(3)))
    plot(f, [-10.0, -10.0, -10.0], 10.0, .1, "exp")
    plt.show()

def plot(funcs, min, max, step, title):
    x = []
    y = []
    for f in funcs:
        i = np.array(min)
        while i[0] < max:
            xk = i
            x.append(xk[0])
            y.append(f['f'](xk))
            i += step
    # print(x)
    print(y)
    plt.yscale("log")
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.ylim(-10e6, 100)
    plt.xlim(min[0], max)
    plt.title(title)
    plt.grid()

if __name__ == "__main__":
    main()