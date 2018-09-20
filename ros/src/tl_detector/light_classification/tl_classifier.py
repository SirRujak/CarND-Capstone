# from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
from tensorflow.contrib.layers import flatten
import numpy as np

WIDTH, HEIGHT, CHANNEL = 224, 224, 3
NUM_CLASS = 4
IMAGE_MEAN = [99.24874878, 97.38613129, 86.80349731]

class TLClassifier(object):

    x = tf.placeholder(tf.float32, (None, HEIGHT, WIDTH, CHANNEL))
    keep_prob = tf.placeholder(tf.float32)
    logit = tf.placeholder(tf.float32, (1, NUM_CLASS))

    def __init__(self):
        self.logit = self.alexnet()
        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, 'model/model.ckpt-640')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # image preprocessing
        img_norm = cv2.resize((image - np.mean(IMAGE_MEAN)), (HEIGHT, WIDTH)) / 256

        # inference
        probs = self.sess.run(tf.nn.softmax(self.logit), feed_dict={self.x: img_norm.reshape(1, WIDTH, HEIGHT, CHANNEL), self.keep_prob: 1.})
        idx = np.argmax(probs)
        return 4 if idx == 3 else idx

    def alexnet(self):
        mu = 0
        sigma = 0.1

        # weights initialization
        conv1_w = tf.Variable(tf.truncated_normal(shape=(11, 11, 3, 64), mean=mu, stddev=sigma))
        conv2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 192), mean=mu, stddev=sigma))
        conv3_w = tf.Variable(tf.truncated_normal(shape=(3, 3, 192, 384), mean=mu, stddev=sigma))
        conv4_w = tf.Variable(tf.truncated_normal(shape=(3, 3, 384, 256), mean=mu, stddev=sigma))
        conv5_w = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 64), mean=mu, stddev=sigma))
        fc1_w = tf.Variable(tf.truncated_normal(shape=(3136, 1024), mean=mu, stddev=sigma))
        fc2_w = tf.Variable(tf.truncated_normal(shape=(1024, 200), mean=mu, stddev=sigma))
        out_w = tf.Variable(tf.truncated_normal(shape=(200, NUM_CLASS), mean=mu, stddev=sigma))

        # bias initialization
        conv1_b = tf.zeros(64)
        conv2_b = tf.zeros(192)
        conv3_b = tf.zeros(384)
        conv4_b = tf.zeros(256)
        conv5_b = tf.zeros(64)
        fc1_b = tf.zeros(1024)
        fc2_b = tf.zeros(200)
        out_b = tf.zeros(NUM_CLASS)

        # conv1
        # input 224x224x3, output 56x56x64
        conv1 = tf.nn.conv2d(self.x, conv1_w, strides=[1, 4, 4, 1], padding='SAME') + conv1_b
        # pooling
        # input 56x56x64, output 28x28x64
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # activation
        activ1 = tf.nn.relu(pool1)

        # conv2
        # input 28x28x64, output 28x28x192
        conv2 = tf.nn.conv2d(activ1, conv2_w, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
        # pooling
        # input 28x28x192, output 14x14x192
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # activation
        activ2 = tf.nn.relu(pool2)

        # conv3
        # input 14x14x192, output 14x14x384
        conv3 = tf.nn.conv2d(activ2, conv3_w, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
        # activation
        activ3 = tf.nn.relu(conv3)

        # conv4
        # input 14x14x384, output 14x14x256
        conv4 = tf.nn.conv2d(activ3, conv4_w, strides=[1, 1, 1, 1], padding='SAME') + conv4_b
        # activation
        activ4 = tf.nn.relu(conv4)

        # conv5
        # input 14x14x256, output 14x14x64
        conv5 = tf.nn.conv2d(activ4, conv5_w, strides=[1, 1, 1, 1], padding='SAME') + conv5_b
        # pooling
        # input 14x14x64, output 7x7x64
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME') + conv5_b
        # activation
        activ5 = tf.nn.relu(pool5)

        # faltten
        fc0 = flatten(activ5)

        # fully-connected 1
        # output 1024
        fc1 = tf.matmul(fc0, fc1_w) + fc1_b
        fc1 = tf.nn.relu(fc1)
        fc1_dropout = tf.nn.dropout(fc1, keep_prob=self.keep_prob)
        # fully-connected 2
        # output 200
        fc2 = tf.matmul(fc1_dropout, fc2_w) + fc2_b
        fc2 = tf.nn.relu(fc2)
        fc2_dropout = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
        # output
        # output num_class
        logits = tf.matmul(fc2_dropout, out_w) + out_b

        return logits

