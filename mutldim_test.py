import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from mine_estimator import mine, DistributionSimulator


def gen_x(data_size):
    return np.random.normal(1.,1.,[data_size[0],data_size[1]])

def gen_y(x, data_size):
    y = 2*x[:, 0] + 0.5*np.square(x[:, 1])*x[:, -1] + np.random.normal(0.1, np.sqrt(2),[data_size,])
    return y



if __name__ == "__main__":
    x = gen_x([10000, 3])
    print('-'*100)
    y = gen_y(x, 10000)
    est, hist = mine(x, y, stop_wait=150)
    print(f'MINE output {est}')
    plt.plot(np.arange(len(hist)), np.array(hist), label='MINE estimation')
    plt.legend()
    plt.show()