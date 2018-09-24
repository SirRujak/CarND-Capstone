# from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
from tensorflow.contrib.layers import flatten
import numpy as np

WIDTH, HEIGHT, CHANNEL = 600, 800, 3
NUM_CLASS = 4
#IMAGE_MEAN = [99.24874878, 97.38613129, 86.80349731]

class TLClassifier(object):

    #x = tf.placeholder(tf.float32, (None, HEIGHT, WIDTH, CHANNEL))
    #keep_prob = tf.placeholder(tf.float32)
    #logit = tf.placeholder(tf.float32, (1, NUM_CLASS))

    def __init__(self):
        #self.logit = self.alexnet()
        self.sess = tf.Session()
        self.model = tf.contrib.keras.models.load_model('model/keras_light_model_130_0.993489583333.h5')
        #tf.train.Saver().restore(self.sess, 'model/model.ckpt-640')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # image preprocessing
        img_norm = image / 255.0

        # inference
        #probs = self.sess.run(tf.nn.softmax(self.logit), feed_dict={self.x: img_norm.reshape(1, WIDTH, HEIGHT, CHANNEL), self.keep_prob: 1.})
        probs = self.model.predict(img_norm.reshape(1, WIDTH, HEIGHT, CHANNEL))
        idx = np.argmax(probs)
        return 4 if idx == 3 else idx
