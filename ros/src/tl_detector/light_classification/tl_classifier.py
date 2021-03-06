# from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
from tensorflow.contrib.layers import flatten
import numpy as np

WIDTH, HEIGHT, CHANNEL = 600, 800, 3
NUM_CLASS = 4

class TLClassifier(object):

    def __init__(self):
        self.sess = tf.Session()
        self.model = tf.contrib.keras.models.load_model('model/keras_light_model_135_0.990301724138.h5')
        
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # image preprocessing
        img_norm = cv2.resize(image, (HEIGHT, WIDTH))
        img_norm = img_norm / 255.0


        # inference
        with self.graph.as_default():
            probs = self.model.predict(np.array(img_norm.reshape(1, WIDTH, HEIGHT, CHANNEL)))
        idx = np.argmax(probs)
        return 4 if idx == 3 else idx
