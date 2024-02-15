import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_image(filename):
    img = Image.open(filename)
    img = np.asarray(img).astype(np.float32)
    img /= 255.0
    return img

def load_data(images_list, labels_list, img_path):
    images = []
    labels = []
    for filename, label in zip(images_list, labels_list):
        images.append(load_image(img_path + filename))
        labels.append(label)
    return images, labels

def preprocess_data(images, labels):
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.15, random_state=666)
    train_images = np.reshape(train_images, (-1,227,227,1))
    train_labels = np.reshape(train_labels, (-1,8))
    test_images = np.reshape(test_images,(-1,227,227,1))
    test_labels = np.reshape(test_labels,(-1,8))
    return train_images, test_images, train_labels, test_labels
