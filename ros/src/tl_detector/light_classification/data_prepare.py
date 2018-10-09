import os
import glob
import cv2
import pickle

data_path = 'data/simulator'
class_name = {'red': 0, 'yellow': 1, 'green': 2, 'none': 3}

images = []
label = []

current_image = 0
for state in class_name.keys():
    for img in glob.glob(os.path.join(data_path, state) + '/*.jpg'):
        images.append(img)
        label.append(class_name[state])
        current_image += 1
        print(current_image)

data = {'image': images, 'label': label}
with open(data_path + '.pkl', 'wb') as f:
    pickle.dump(data, f)
