import csv
import os

import cv2 as cv
import glob
import numpy
from keras import Sequential, Model
from keras.layers import ZeroPadding2D, Convolution2D, Dropout, Flatten, Activation, MaxPooling2D
from tqdm import tqdm
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from comparator import *

def make_row(person, param, symptoms):
    return {'name': person,
            'frame': param,
            'symptoms': symptoms
            }


def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model


def make_symptoms(only_average):
    # model = create_model()
    # model.load_weights('vgg_face_weights.h5')
    # descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    path = 'resource/csv_symptoms_resnet_best/'
    descriptor = ResNet50(weights="imagenet", include_top=False)
    for t_f_pairs in ('resource/TruePairs', 'resource/FalsePairs'):
        pairs = os.listdir(t_f_pairs)
        for pair in tqdm(pairs):
            persons = os.listdir(t_f_pairs + '/' + pair)
            persons = [x for x in persons if ".png" not in x]
            for person in persons:
                if not os.path.exists(path + person + '.csv'):
                    with open((path + person + '.csv'), 'w', encoding='UTF8', newline='') as f:
                        header = ['name', 'frame', 'symptoms']
                        writer = csv.DictWriter(f, fieldnames=header)
                        writer.writeheader()

                        # image = cv.imread(t_f_pairs + '/' + pair + '/' + person + '.png')
                        # image = cv.resize(image, (224, 224))
                        image = load_img(t_f_pairs + '/' + pair + '/' + person + '.png', target_size=(64, 64))
                        img_input = preprocess_input(numpy.expand_dims(img_to_array(image), axis=0))
                        symptoms = descriptor.predict(img_input).ravel().tolist()
                        writer.writerow(make_row(person, 'average', symptoms))
                        if not only_average:
                            images = glob.glob(t_f_pairs + '/' + pair + '/' + person + '/*.png')
                            for img_path in images:
                                # image = cv.imread(img_path)
                                # image = cv.resize(image, (224, 224))
                                image = load_img(img_path, target_size=(64, 64))
                                img_input = preprocess_input(numpy.expand_dims(img_to_array(image), axis=0))
                                symptoms = descriptor.predict(img_input).ravel().tolist()
                                writer.writerow(make_row(person, os.path.basename(img_path)[:-4], symptoms))


def main_vgg_face():
    make_symptoms(True)
    vgg_face_compare_symptoms_average_faces('resource/euclidean_distances/best/average_faces-')

    # make_symptoms(False)
    # vgg_face_compare_symptoms_average_faces('resource/euclidean_distances/resnet/average_faces-')
    # vgg_face_compare_symptoms_random_frame()
    # vgg_face_compare_symptoms_all_frames_average()

