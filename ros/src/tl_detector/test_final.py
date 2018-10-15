import tensorflow as tf
import glob, os
import cv2
import numpy as np

WIDTH, HEIGHT, CHANNEL = 600, 800, 3
NUM_CLASS = 4
model = tf.contrib.keras.models.load_model('model/keras_light_model_135_0.990301724138.h5')

os.chdir("./data/simulator")

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
for file in glob.glob("*/*.jpg"):
    img = cv2.imread(file)
    img = cv2.resize(img, (HEIGHT, WIDTH))
    probs = model.predict(np.array(np.expand_dims(img, axis=0)))
    idx = np.argmax(probs)
    prediction = None
    if idx == 2 and idx == 3:
        print(idx)
    if idx == 0:
        prediction = "RED"
    elif idx == 1:
        prediction = "YELLOW"
    elif idx == 2:
        prediction = "GREEN"
    else:
        prediction = "NO_LIGHT"
    print(prediction)
    filename = file.split("/")[1]
    cv2.imwrite("new_imgs/" + prediction + '/' + filename, img)


    


