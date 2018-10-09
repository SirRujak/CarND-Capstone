import pickle
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import cv2

# define image input size
HEIGHT = 224
WIDTH = 224
num_channel = 3
num_class = 4

# training parameters setting
epoch = 500
batch_size = 32
learning_rate = 0.000001

split = 0.9


def preprocess(X):
    # # convert from RGB to gray
    # X_lab = np.array([cv2.cvtColor(rgb_img, cv2.COLOR_BGR2Lab) for rgb_img in X], np.float32)
    image_mean = np.array([np.mean(X[:, :, :, c]) for c in range(X.shape[-1])], np.float32)

    # normalization
    X_norm = np.array([cv2.resize((img - np.mean(image_mean)), (HEIGHT, WIDTH)) / 256 for img in X])

    return X_norm
'''

def AlexNet():
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
    out_w = tf.Variable(tf.truncated_normal(shape=(200, num_class), mean=mu, stddev=sigma))

    # bias initialization
    conv1_b = tf.zeros(64)
    conv2_b = tf.zeros(192)
    conv3_b = tf.zeros(384)
    conv4_b = tf.zeros(256)
    conv5_b = tf.zeros(64)
    fc1_b = tf.zeros(1024)
    fc2_b = tf.zeros(200)
    out_b = tf.zeros(num_class)

    # conv1
    # input 224x224x3, output 56x56x64
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 4, 4, 1], padding='SAME') + conv1_b
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
    fc1_dropout = tf.nn.dropout(fc1, keep_prob=keep_prob)
    # fully-connected 2
    # output 200
    fc2 = tf.matmul(fc1_dropout, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)
    fc2_dropout = tf.nn.dropout(fc2, keep_prob=keep_prob)
    # output
    # output num_class
    logits = tf.matmul(fc2_dropout, out_w) + out_b

    print('input: ', x)
    print('conv1: ', conv1.shape)
    print('pool1: ', pool1.shape)
    print('conv2: ', conv2.shape)
    print('pool2: ', pool2.shape)
    print('conv3: ', conv3.shape)
    print('conv4: ', conv4.shape)
    print('conv5: ', conv5.shape)
    print('fc1: ', fc1.shape)
    print('fc2: ', fc2.shape)
    print('output: ', logits.shape)
    return logits


def LightNet(num_channel=3, num_class=4):
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
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
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
    fc1_dropout = tf.nn.dropout(fc1, keep_prob=keep_prob)
    # fully-connected 2
    # output 500
    fc2 = tf.matmul(fc1_dropout, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)
    fc2_dropout = tf.nn.dropout(fc2, keep_prob=keep_prob)
    # output
    # output 4
    logits = tf.matmul(fc2_dropout, out_w) + out_b

    # print out network architecture
    print('input: ', x)
    print('conv1: ', conv1.shape)
    print('pool1: ', pool1.shape)
    print('conv2: ', conv2.shape)
    print('pool2: ', pool2.shape)
    print('conv3: ', conv3.shape)
    print('pool2: ', pool3.shape)
    print('fc1: ', fc1.shape)
    print('fc2: ', fc2.shape)
    print('output: ', logits.shape)

    return logits
'''
'''
input_tensor = tf.contrib.keras.layers.Input(shape=(WIDTH, HEIGHT,3,))
model = tf.contrib.keras.applications.InceptionV3(input_tensor=input_tensor, include_top=False)
x = model.output
model = tf.contrib.keras.layers.Dense(4)(x)
model = tf.contrib.keras.models.Model([input_tensor], [model])
'''
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
        ##print(img_data)
        batch_data.append(img_data/255.0)
        temp_label = label_list[key]
        if temp_label == 4:
            temp_label = 3
        batch_labels.append(temp_label)
        #print(temp_label)
        #print(tf.contrib.keras.utils.to_categorical(np.array(temp_label), num_classes=4))
    return np.array(batch_data), tf.contrib.keras.utils.to_categorical(np.array(batch_labels), num_classes=4)

def train(data, label):
    # define operations
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    #loss_operation = tf.reduce_mean(cross_entropy)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #training_operation = optimizer.minimize(loss_operation)
    #saver = tf.train.Saver()
    train_data = data[:int(len(data) * split)]
    train_label = label[:int(len(data) * split)]
    validation_data = data[int(len(data) * split):]
    validation_label = label[int(len(data) * split):]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Start training ...\n')
        for e in range(epoch):
            current_image = 0
            for start in range(0, len(train_data), batch_size):
                end = start + batch_size
                #x_batch, y_batch = data[start:end], label[start:end]
                x_batch, y_batch = get_batch(train_data[start:end], train_label[start:end])
                model.train_on_batch(x_batch, y_batch)
                #sess.run(training_operation, feed_dict={x: x_batch, y: y_batch, keep_prob: 0.5})
                if current_image % 3 == 0:
                    print("Processing image: " + str(current_image * batch_size))
                current_image += 1
            validation_accuracy = evaluate(validation_data, validation_label)
            print("epoch ", e + 1)
            print("Validation accuracy = {:.3f}\n".format(validation_accuracy))
            train_data, train_label = shuffle(train_data, train_label)
            if e % 10 == 0:
                #saver.save(sess, './model/model.ckpt', global_step=e)
                model.save('keras_light_model_' + str(e) + '_' + str(validation_accuracy) + '.h5')
        print("Model Saved")


def evaluate(data, label):
    # define operations
    #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    #accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_examples = len(data)
    total_accuracy = 0
    sess = tf.get_default_session()
    num_batches = 0
    for offset in range(0, num_examples, batch_size):
        #batch_x, batch_y = data[offset:offset+batch_size], label[offset:offset+batch_size]
        batch_x, batch_y = get_batch(data[offset:offset+batch_size], label[offset:offset+batch_size])
        #accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
        #accuracy = model.evaluate(batch_x, batch_y)
        #prediction = model.predict(batch_x)
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

# prepare input data
x = tf.placeholder(tf.float32, (None, HEIGHT, WIDTH, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, num_class)
keep_prob = tf.placeholder(tf.float32)

# load data
data_path = 'data/simulator.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

images = np.array(data['image'])
labels = np.array(data['label'])
#images = preprocess(images)

# shuffle data
data, label = shuffle(images, labels)

# train
#logits = AlexNet()
train(data, label)
