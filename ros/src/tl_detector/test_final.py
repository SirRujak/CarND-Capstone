import tensorflow as tf
import glob, os
import cv2
import numpy as np

WIDTH, HEIGHT, CHANNEL = 600, 800, 3
NUM_CLASS = 4
model = tf.contrib.keras.models.load_model('model/keras_light_model_130_0.993489583333.h5')

os.chdir("./data/simulator")

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

for file in glob.glob("*.jpg"):
    ## Load file
    img = cv2.imread(file)
    img = cv2.resize(img, (HEIGHT, WIDTH))
    probs = model.predict(np.array(img.reshape(1, WIDTH, HEIGHT, CHANNEL)))
    idx = np.argmax(probs)
    if idx == 0:
        prediction = "RED"
    if idx == 1:
        prediction = "YELLOW"
    if idx == 2:
        prediction = "GREEN"
    else:
        prediction = "NO LIGHT"

    cv2.putText(img, prediction,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)
    cv2.imwrite("new_imgs/" + file, img)


    


