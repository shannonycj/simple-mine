import tensorflow as tf
import numpy as np


class DistributionSimulator:
    
    def __init__(self, x, y):
        self.x = np.array(x); self.y = np.array(y)
    
    def reshuffle(self):
        y_cpy = self.y.copy()
        pool = np.array([self.x, self.y]).T
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


def mine(x, y, n_hidden=50, lr=0.1, batch_size=128, early_stopping=40):
    ds = DistributionSimulator(x, y)
    initializer = tf.glorot_normal_initializer()
    xy_in = tf.placeholder(tf.float32, shape=[None, 2])
    xy_bar_in = tf.placeholder(tf.float32, shape=[None, 2])
    W_1 = tf.Variable(initializer([2, n_hidden]), dtype=tf.float32)
    b_1 = tf.Variable(tf.zeros(n_hidden), dtype=tf.float32)
    z_1 = tf.matmul(xy_in, W_1) + b_1
    z_1_bar = tf.matmul(xy_bar_in, W_1) + b_1
    a_1 = z_1*tf.nn.sigmoid(z_1)
    a_1_bar = z_1_bar*tf.nn.sigmoid(z_1_bar)

    W_2 = tf.Variable(initializer([n_hidden, 1]), dtype=tf.float32)
    b_2 = tf.Variable(tf.zeros(1), dtype=tf.float32)
    z_2 = tf.matmul(a_1, W_2) + b_2
    z_2_bar = tf.matmul(a_1_bar, W_2) + b_2
    a_2 = z_2*tf.nn.sigmoid(z_2)
    a_2_bar = z_2_bar*tf.nn.sigmoid(z_2_bar)

    neg_loss = -(tf.reduce_mean(a_2, axis=0) - tf.math.log(tf.reduce_mean(tf.math.exp(a_2_bar), axis=0)))
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(neg_loss)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    training_loss = []
    for epoch in range(1000):
        ds.reshuffle()
        ds.init_batches(batch_size)
        done = False
        losses = []
        while not done:
            batch, done = ds.next_batch()
            batch_xy = batch[:, :2]
            batch_x_y = np.array([batch[:, 0], batch[:, -1]]).T
            _, loss = sess.run([opt, neg_loss], feed_dict={xy_in: batch_xy, xy_bar_in: batch_x_y})
            losses.append(loss)
        if epoch > 1.5*early_stopping:
            if loss >= np.max(training_loss[-early_stopping:]):
                break
        print(f'epoch {epoch}, loss {np.mean(losses)}')
        training_loss.append(np.mean(losses))
    sess.close()
    return -np.min(training_loss[-early_stopping:]), training_loss