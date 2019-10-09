import tensorflow as tf
import numpy as np


class DistributionSimulator:
    
    def __init__(self, x, y):
        self.x = np.array(x); self.y = np.array(y)
        self.standardize()
    
    def standardize(self):
        self.x = (self.x - self.x.mean(axis=0, keepdims=True))/self.x.std(axis=0, keepdims=True)
        #self.x = (self.x - self.x.mean())/self.x.std()
        self.y = (self.y - self.y.mean())/self.y.std()

    def reshuffle(self):
        y_cpy = self.y.copy()
        pool = np.concatenate([self.x, self.y.reshape(-1, 1)], axis=1)
        #pool = np.array([self.x, self.y]).T
        np.random.shuffle(pool)
        np.random.shuffle(y_cpy)
        self.pool = np.concatenate([pool, y_cpy.reshape(-1, 1)], axis=1)
    
    def init_batches(self, batch_size):
        self.batch_size = batch_size
        self.n_batch = (self.x.shape[0] // batch_size) + ((self.x.shape[0] % batch_size) > 0)*1
        self.batch_num = 0
    
    def next_batch(self):
        end_idx = min((self.batch_num + 1) * self.batch_size, self.pool.shape[0])
        batch = self.pool[self.batch_size*self.batch_num : end_idx]
        self.batch_num += 1
        return batch, end_idx == self.pool.shape[0]


def build_net(n_hidden, lr, global_step, decay_steps, xy_shape=2):
    initializer = tf.variance_scaling_initializer(distribution='uniform')
    xy_in = tf.placeholder(tf.float32, shape=[None, xy_shape])
    xy_bar_in = tf.placeholder(tf.float32, shape=[None, xy_shape])
    W_1 = tf.Variable(initializer([xy_shape, n_hidden]), dtype=tf.float32)
    b_1 = tf.Variable(tf.zeros(n_hidden), dtype=tf.float32)
    z_1 = tf.matmul(xy_in, W_1) + b_1
    z_1_bar = tf.matmul(xy_bar_in, W_1) + b_1
    a_1 = tf.nn.leaky_relu(z_1)
    a_1_bar = tf.nn.leaky_relu(z_1_bar)

    W_2 = tf.Variable(initializer([n_hidden, 1]), dtype=tf.float32)
    b_2 = tf.Variable(tf.zeros(1), dtype=tf.float32)
    z_2 = tf.matmul(a_1, W_2) + b_2
    z_2_bar = tf.matmul(a_1_bar, W_2) + b_2
    a_2 = tf.nn.leaky_relu(z_2)
    a_2_bar = tf.nn.leaky_relu(z_2_bar)

    neural_info_measure = tf.reduce_mean(a_2, axis=0) - tf.math.log(tf.reduce_mean( \
                                tf.math.exp(a_2_bar), axis=0))
    learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, 0.99, staircase=True)
    optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(-neural_info_measure)
    return xy_in, xy_bar_in, neural_info_measure, optimize

def mine(x, y, n_hidden=50, lr=0.05, batch_size=128, early_stopping=40, stop_wait=100):
    ds = DistributionSimulator(x, y)
    ds.init_batches(batch_size)
    xy_shape = ds.x.shape[1] + 1
    global_step = ds.n_batch * 100
    decay_steps = int(global_step / 100)
    xy_in, xy_bar_in, neural_info_measure, optimize = build_net(n_hidden, lr, global_step, decay_steps, xy_shape)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    neural_info_estimates = []
    for epoch in range(1000):
        ds.init_batches(batch_size)
        ds.reshuffle()
        done = False
        batch_mi = []
        while not done:
            batch, done = ds.next_batch()
            batch_xy = batch[:, :-1]
            batch_x_y = np.concatenate([batch[:, :-2], batch[:, -1].reshape(-1,1)], axis=1)
            _, mi = sess.run([optimize, neural_info_measure], feed_dict={xy_in: batch_xy, \
                                                                    xy_bar_in: batch_x_y})
            batch_mi.append(mi)
        if epoch > stop_wait:
            if mi >= np.max(neural_info_estimates[-early_stopping:]):
                break
        print(f'epoch: {epoch}, MI estimation: {np.mean(batch_mi)}')
        neural_info_estimates.append(np.mean(batch_mi))
    sess.close()
    eval_idx = max(int(early_stopping/4), 5)
    return np.mean(neural_info_estimates[-eval_idx:]), neural_info_estimates