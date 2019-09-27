import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from mine_estimator import mine


def func(x):
    return x

def gen_x(data_size):
    return np.sign(np.random.normal(0.,1.,[data_size,1]))

def gen_y(x, data_size):
    return func(x)+np.random.normal(0.,np.sqrt(0.2),[data_size,1])


if __name__ == "__main__":
    x_sample=gen_x(10000)
    y_sample=gen_y(x_sample, 10000)

    res, est_hist = mine(x_sample.reshape(-1, ), y_sample.reshape(-1,))
    mi_numerical = mutual_info_regression(x_sample.reshape(-1, 1), y_sample.reshape(-1,))[0]
    print(f'MINE output {res}')
    print(f'scikit-learn output {mi_numerical}')
    plt.plot(np.arange(len(est_hist)), np.array(est_hist), label='MINE estimation')
    plt.plot(np.arange(len(est_hist)), np.ones(len(est_hist))*mi_numerical, label='True')
    plt.legend()
    plt.show()