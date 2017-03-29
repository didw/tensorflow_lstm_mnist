from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
from etaprogress.progress import ProgressBar
import tensorflow as tf
import numpy as np
import sys, os, shutil
import time


class TensorflowLSTM():
    def __init__(self, h_size=128, n_inputs=28, n_steps=28, n_classes=10, l_r=0.001, batch_size=128):
        # parameters init
        l_r = l_r
        self.n_inputs = n_inputs
        self.n_steps = n_steps
        n_classes = n_classes
        self.model_dir = 'model/tf/lstm'

        ## build graph
        tf.reset_default_graph()
        self.batch_size = tf.placeholder(tf.int32)
        self.X = tf.placeholder(tf.float32, shape=(None, n_steps, n_inputs))
        self.Y = tf.placeholder(tf.float32, [None, n_classes])

        w1 = tf.Variable(tf.random_normal([h_size, n_classes]))
        b1 = tf.Variable(tf.random_normal([n_classes]))

        cell = tf.contrib.rnn.BasicLSTMCell(h_size)
        time = tf.constant(0, dtype=tf.int32)
        zero_state = tf.zeros([self.batch_size, n_inputs], dtype=tf.float32)

        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output  # == None for time == 0
            if cell_output is None:  # time == 0
                next_cell_state = cell.zero_state(self.batch_size, tf.float32)
            else:
                next_cell_state = cell_state
            elements_finished = (time >= n_steps)
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                finished,
                lambda: zero_state,
                lambda: self.X[:,time])
            next_loop_state = None
            time += 1
            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        outputs = outputs_ta.stack()

        self.pred = tf.matmul(outputs[-1,:], w1) + b1
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.Y))
        tf.summary.scalar('loss', self.cost)
        self.train_op = tf.train.AdamOptimizer(l_r).minimize(self.cost)

        correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def fit(self, mnist, n_epoch=10, batch_size=128):
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        test_x = mnist.test.images.reshape([-1, self.n_steps, self.n_inputs])
        test_y = mnist.test.labels
        acc = 0
        with tf.Session(config=config) as sess:
            shutil.rmtree('/tmp/TF/MNIST')
            self.sw_train = tf.summary.FileWriter('/tmp/TF/MNIST/train', sess.graph)
            self.sw_test = tf.summary.FileWriter('/tmp/TF/MNIST/test')
            sess.run(init_op)
            for i in range(n_epoch):
                print('Epoch %d/%d' % (i+1, n_epoch))
                bar = ProgressBar(int(60000/batch_size), max_width=80)
                for j in range(int(60000/batch_size)):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    batch_x = batch_x.reshape([batch_size, self.n_steps, self.n_inputs])
                    summary, _, cost = sess.run([self.merged, self.train_op, self.cost], feed_dict={self.X: batch_x, self.Y: batch_y, self.batch_size: 128})
                    self.sw_train.add_summary(summary, i*int(60000/batch_size) + j)
                    bar.numerator = j+1
                    print("%s | loss: %f | test_acc: %.2f" % (bar, cost, acc*100), end='\r')
                    sys.stdout.flush()
                    if j % 100 == 0:
                        summary, cost, acc = sess.run([self.merged, self.cost, self.accuracy], feed_dict={self.X: test_x, self.Y: test_y, self.batch_size: 10000})
                        self.sw_test.add_summary(summary, i*int(60000/batch_size)+j)
                print()
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            saver = tf.train.Saver()
            save_path = saver.save(sess,'%s/model.ckpt' % self.model_dir)
            print("Model saved in file: %s" % save_path)

    def predict(self, X_test):
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            return sess.run(self.pred, feed_dict={self.X: X_test, self.batch_size: 10000})

def main():
    #load mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    tf_lstm = TensorflowLSTM()
    t1 = time.time()
    tf_lstm.fit(mnist)
    t2 = time.time()
    print('training time: %s' % (t2-t1))
    pred = tf_lstm.predict(mnist.test.images.reshape(-1, 28, 28))
    t3 = time.time()
    print('predict time: %s' % (t3-t2))
    test_lab = mnist.test.labels
    print("accuracy: ", np.mean(np.equal(np.argmax(pred,1), np.argmax(test_lab,1)))*100)


if __name__ == '__main__':
    main()
