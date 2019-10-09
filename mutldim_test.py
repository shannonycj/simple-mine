import numpy as np
# import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from mine_estimator import mine, DistributionSimulator


def gen_x(data_size):
    return np.random.normal(1.,1.,[data_size[0],data_size[1]])

def gen_y(x, data_size):
    return 2*x[:, 0] + 0.5*np.square(x[:, 1]) + np.random.normal(0.1,np.sqrt(5),[data_size,])



if __name__ == "__main__":
    x = gen_x([10000, 3])
    print('-'*100)
    y = gen_y(x, 10000)
    # print(x)
    # print((x - x.mean(axis=0, keepdims=True))/x.std(axis=0, keepdims=True))
    # print(y)
    mine(x, y)
    # ds = DistributionSimulator(x, y)
    # ds.init_batches(10)
    # ds.reshuffle()
    # print(ds.next_batch())
    # mi_numerical = mutual_info_classif(x_sample.reshape(-1, 1), y_sample.reshape(-1,))[0]
    # print(f'MINE output {res}')
    # print(f'scikit-learn output {mi_numerical}')
    # plt.plot(np.arange(len(est_hist)), np.array(est_hist), label='MINE estimation')
    # plt.plot(np.arange(len(est_hist)), np.ones(len(est_hist))*mi_numerical, label='True')
    # plt.legend()
    # plt.show()