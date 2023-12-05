import copy
import numpy as np
import h5py
import pickle
from PIL import Image
import os
import glob

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    print("train_set_x_orig.shape: " + str(train_set_x_orig.shape))
    print("train_set_y_orig.shape: " + str(train_set_y_orig.shape))
    print(classes)

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def img_to_matrix(img_file_path, dim):
    image = Image.open(img_file_path)
    resized_image = image.resize((dim, dim))
    image_array = np.array(resized_image).reshape(dim, dim, -1)

    return image_array

def load_img_files(folder_path, dim=100):
    jpg_files = glob.glob(os.path.join(folder_path, '*.jpg'), recursive=False)

    d = []
    i = 0
    for jpg_file in jpg_files:
        i += 1
        if i > 400:
            continue
            pass
        img = img_to_matrix(jpg_file, dim)
        d.append(img)

    return d

if __name__ == '__main__':
    load_dataset()
    dim = 64

    print("Creating training set...")
    train_set_x_pos = load_img_files('images/train/cats', dim)
    train_set_x_neg = load_img_files('images/train/dogs', dim)
    train_set_x = train_set_x_pos + train_set_x_neg
    train_set_x = np.array(train_set_x)

    print("train_set_x.shape: " + str(train_set_x.shape))
    print("len(train_set_x_pos): " + str(len(train_set_x_pos)))
    print("len(train_set_x_neg): " + str(len(train_set_x_neg)))

    train_set_y_pos = np.ones((len(train_set_x_pos)))
    train_set_y_neg = np.zeros((len(train_set_x_neg)))

    print("train_set_y_pos.shape: " + str(train_set_y_pos.shape))
    print("train_set_y_neg.shape: " + str(train_set_y_neg.shape))

    train_set_y = np.append(train_set_y_pos, train_set_y_neg)

    print("train_set_y.shape: " + str(train_set_y.shape))

    with h5py.File('datasets/train_cats.h5', 'w') as file:
        file.create_dataset('train_set_x', data=train_set_x)
        file.create_dataset('train_set_y', data=train_set_y)


    print("Creating test set...")
    test_set_x = load_img_files('images/test/cats', dim)
    test_set_x = np.array(test_set_x)

    print("test_set_x.shape: " + str(test_set_x.shape))

    test_set_y = np.ones((test_set_x.shape[0]))
    list_classes = ['non-cat', 'cat']

    print("test_set_y.shape: " + str(test_set_y.shape))

    with h5py.File('datasets/test_cats.h5', 'w') as file:
        file.create_dataset('test_set_x', data=test_set_x)
        file.create_dataset('test_set_y', data=test_set_y)
        file.create_dataset('list_classes', data=list_classes)

    train_dataset = h5py.File('datasets/train_cats.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    print("Loading data back...")
    print("train_set_x_orig.shape: " + str(train_set_x_orig.shape))
    print("train_set_y_orig.shape: " + str(train_set_y_orig.shape))