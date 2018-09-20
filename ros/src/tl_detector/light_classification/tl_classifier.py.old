from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
from tensorflow.contrib.layers import flatten
import numpy as np

WIDTH, HEIGHT, CHANNEL = 32, 32, 3
NUM_CLASS = 4

class TLClassifier(object):

    x = tf.placeholder(tf.float32, (None, HEIGHT, WIDTH, CHANNEL))
    keep_prob = tf.placeholder(tf.float32)
    logit = tf.placeholder(tf.float32, (1, NUM_CLASS))

    def __init__(self):
        self.logit = self.net()
        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, 'model/model.ckpt-100')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        resized_img = cv2.resize(image, (WIDTH, HEIGHT))
        probs = self.sess.run(tf.nn.softmax(self.logit), feed_dict={self.x: resized_img.reshape(1, WIDTH, HEIGHT, CHANNEL), self.keep_prob: 1.})
        idx = np.argmax(probs)
        return 4 if idx == 3 else idx

    def net(self, num_channel=3, num_class=4):
        mu = 0
        sigma = 0.1

        # weights initialization
        conv1_w = tf.Variable(tf.truncated_normal(shape=(3, 3, num_channel, 64), mean=mu, stddev=sigma))
        conv2_w = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean=mu, stddev=sigma))
        conv3_w = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 64), mean=mu, stddev=sigma))
        fc1_w = tf.Variable(tf.truncated_normal(shape=(1024, 1024), mean=mu, stddev=sigma))
        fc2_w = tf.Variable(tf.truncated_normal(shape=(1024, 500), mean=mu, stddev=sigma))
        out_w = tf.Variable(tf.truncated_normal(shape=(500, num_class), mean=mu, stddev=sigma))

        # bias initialization
        conv1_b = tf.zeros(64)
        conv2_b = tf.zeros(128)
        conv3_b = tf.zeros(64)
        fc1_b = tf.zeros(1024)
        fc2_b = tf.zeros(500)
        out_b = tf.zeros(num_class)

        # conv1
        # input 32x32x3, output 32x32x64
        conv1 = tf.nn.conv2d(self.x, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
        # pooling
        # input 32x32x64, output 16x16x64
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # activation
        activ1 = tf.nn.relu(pool1)

        # conv2
        # input 16x16x64, output 16x16x128
        conv2 = tf.nn.conv2d(activ1, conv2_w, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
        # pooling
        # input 16x16x128, output 8x8x128
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # activation
        activ2 = tf.nn.relu(pool2)

        # conv3
        # input 8x8x128, output 8x8x64
        conv3 = tf.nn.conv2d(activ2, conv3_w, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
        # pooling
        # intput 8x8x64, output 4x4x64
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # activation
        # output 1024
        activ3 = tf.nn.relu(pool3)

        # faltten
        fc0 = flatten(activ3)

        # fully-connected 1
        # output 1024
        fc1 = tf.matmul(fc0, fc1_w) + fc1_b
        fc1 = tf.nn.relu(fc1)
        fc1_dropout = tf.nn.dropout(fc1, keep_prob=self.keep_prob)
        # fully-connected 2
        # output 500
        fc2 = tf.matmul(fc1_dropout, fc2_w) + fc2_b
        fc2 = tf.nn.relu(fc2)
        fc2_dropout = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
        # output
        # output 4
        logits = tf.matmul(fc2_dropout, out_w) + out_b

        return logits
