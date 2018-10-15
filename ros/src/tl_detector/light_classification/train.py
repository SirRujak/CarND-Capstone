import pickle
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import cv2

# define image input size
HEIGHT = 800
WIDTH = 600
num_channel = 3
num_class = 4

# training parameters setting
epoch = 500
batch_size = 32
learning_rate = 0.000001

split = 0.9

model = tf.contrib.keras.models.Sequential()
model.add(tf.contrib.keras.layers.Conv2D(8, 4, 4, activation='relu', padding="valid", batch_input_shape=(None, 600, 800, 3,)))
model.add(tf.contrib.keras.layers.Conv2D(32, 3, 3, activation='relu', padding="valid"))
model.add(tf.contrib.keras.layers.Conv2D(64, 2, 2, activation='relu', padding="valid"))
model.add(tf.contrib.keras.layers.Conv2D(96, 2, 2, activation='relu', padding="valid"))
model.add(tf.contrib.keras.layers.Conv2D(128, 2, 2, activation='relu', padding="valid"))
model.add(tf.contrib.keras.layers.Flatten())
model.add(tf.contrib.keras.layers.Dense(256))
model.add(tf.contrib.keras.layers.Dense(128))
model.add(tf.contrib.keras.layers.Dense(64))
model.add(tf.contrib.keras.layers.Dense(32))
model.add(tf.contrib.keras.layers.Dense(4))
model.add(tf.contrib.keras.layers.Activation('softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['mse', 'accuracy'])

def get_batch(data_list, label_list):
    batch_labels = []
    batch_data = []
    for key, file_name in enumerate(data_list):
        img_data = cv2.imread(file_name)
        img_data = cv2.resize(img_data, (HEIGHT, WIDTH))
        batch_data.append(img_data/255.0)
        temp_label = label_list[key]
        if temp_label == 4:
            temp_label = 3
        batch_labels.append(temp_label)
    return np.array(batch_data), tf.contrib.keras.utils.to_categorical(np.array(batch_labels), num_classes=4)

def train(data, label):
    train_data = data[:int(len(data) * split)]
    print(len(train_data))
    train_label = label[:int(len(data) * split)]
    validation_data = data[int(len(data) * split):]
    validation_label = label[int(len(data) * split):]
    highest_accuracy = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Start training ...\n')
        for e in range(epoch):
            current_image = 0
            for start in range(0, len(train_data), batch_size):
                end = start + batch_size
                x_batch, y_batch = get_batch(train_data[start:end], train_label[start:end])
                model.train_on_batch(x_batch, y_batch)
                if current_image % 3 == 0:
                    print("Processing image: " + str(current_image * batch_size))
                current_image += 1
            validation_accuracy = evaluate(validation_data, validation_label)
            print("epoch ", e + 1)
            print("Validation accuracy = {:.3f}\n".format(validation_accuracy))
            train_data, train_label = shuffle(train_data, train_label)
            if validation_accuracy > highest_accuracy:
                highest_accuracy = validation_accuracy
                model.save('keras_light_model' + str(e) + '_' + str(validation_accuracy) + '.h5')
        print("Model Saved")


def evaluate(data, label):

    num_examples = len(data)
    total_accuracy = 0
    sess = tf.get_default_session()
    num_batches = 0
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = get_batch(data[offset:offset+batch_size], label[offset:offset+batch_size])
        loss, mse, accuracy = model.evaluate(batch_x, batch_y)
        print(accuracy)
        total_accuracy += accuracy
        num_batches += 1
    return total_accuracy / num_batches

# -------------------------------------------------------------
#
#                            Entrance
#
#--------------------------------------------------------------

# load data
data_path = 'data/simulator.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

images = np.array(data['image'])
labels = np.array(data['label'])

# shuffle data
data, label = shuffle(images, labels)

# train
train(data, label)
