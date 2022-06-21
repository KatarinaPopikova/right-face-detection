import sys

import numpy
import os
import csv
import cv2 as cv
import glob
import numpy as np
import random

from tqdm import tqdm


def compare_symptoms_average_faces(eigenfaces, mean_face, csv_path):
    header = ['name1', 'name2', 'pair index', 'euclidean_distance']

    for t_f_pairs in ('resource/TruePairs', 'resource/FalsePairs'):

        with open(csv_path + t_f_pairs[9:] + '.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            pairs = os.listdir(t_f_pairs)
            for pair in pairs:
                images = glob.glob(t_f_pairs + '/' + pair + '/*.png')
                average_faces = []

                for img in images:
                    this_image = cv.imread(img, 1)
                    average_faces.append(this_image)
                    # average_faces.append(cv.cvtColor(this_image, cv.COLOR_BGR2GRAY))

                query_weight1 = eigenfaces @ (average_faces[0].flatten() - mean_face).T
                query_weight2 = eigenfaces @ (average_faces[1].flatten() - mean_face).T
                euclidean_distance = np.linalg.norm(query_weight2 - query_weight1, axis=0)
                row = {'name1': os.path.basename(images[0])[:-4],
                       'name2': os.path.basename(images[1])[:-4],
                       'pair index': pair,
                       'euclidean_distance': euclidean_distance
                       }

                writer.writerow(row)


def compare_symptoms_random_frame(eigenfaces, mean_face):
    csv_path = 'resource/euclidean_distances/pca/random_faces-'

    for t_f_pairs in ('resource/TruePairs', 'resource/FalsePairs'):

        with open(csv_path + t_f_pairs[9:] + '.csv', 'w', encoding='UTF8', newline='') as f:
            header = ['name1', 'name2', 'pair index', 'euclidean_distance']
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            pairs = os.listdir(t_f_pairs)
            for pair in pairs:
                random_faces = []
                names = []
                persons = os.listdir(t_f_pairs + '/' + pair)
                persons = [x for x in persons if ".png" not in x]
                for person in persons:
                    images = glob.glob(t_f_pairs + '/' + pair + '/' + person + '/*.png')
                    img = images[random.randint(0, len(images) - 1)]
                    names.append(os.path.basename(img)[:-4])
                    random_faces.append(cv.cvtColor(cv.imread(img, 1), cv.COLOR_BGR2GRAY))

                query_weight1 = eigenfaces @ (random_faces[0].flatten() - mean_face).T
                query_weight2 = eigenfaces @ (random_faces[1].flatten() - mean_face).T
                euclidean_distance = np.linalg.norm(query_weight2 - query_weight1, axis=0)
                row = {'name1': str(names[0]) + '|| ' + persons[0],
                       'name2': str(names[1]) + '|| ' + persons[1],
                       'pair index': pair,
                       'euclidean_distance': euclidean_distance
                       }

                writer.writerow(row)


def compare_symptoms_all_frames_average(eigenfaces, mean_face):
    csv_path = 'resource/euclidean_distances/pca/all_frames(average)-'

    for t_f_pairs in ('resource/TruePairs', 'resource/FalsePairs'):

        with open(csv_path + t_f_pairs[9:] + '.csv', 'w', encoding='UTF8', newline='') as f:
            header = ['name1', 'name2', 'pair index', 'euclidean_distance']
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            pairs = os.listdir(t_f_pairs)
            for pair in tqdm(pairs):
                persons = os.listdir(t_f_pairs + '/' + pair)
                persons = [x for x in persons if ".png" not in x]
                images1 = glob.glob(t_f_pairs + '/' + pair + '/' + persons[0] + '/*.png')
                images2 = glob.glob(t_f_pairs + '/' + pair + '/' + persons[1] + '/*.png')

                euclidean_distance = 0
                for img1_path in images1:
                    img1 = cv.cvtColor(cv.imread(img1_path, 1), cv.COLOR_BGR2GRAY)

                    for img2_path in images2:
                        img2 = cv.cvtColor(cv.imread(img2_path, 1), cv.COLOR_BGR2GRAY)
                        query_weight1 = eigenfaces @ (img1.flatten() - mean_face).T
                        query_weight2 = eigenfaces @ (img2.flatten() - mean_face).T
                        euclidean_distance += np.linalg.norm(query_weight2 - query_weight1, axis=0) / (
                                len(images1) * len(images2))

                row = {'name1': persons[0],
                       'name2': persons[1],
                       'pair index': pair,
                       'euclidean_distance': euclidean_distance
                       }

                writer.writerow(row)


def read_average_symptoms(file_name, column):
    csv.field_size_limit(2000000)

    # reader = csv.reader(open('resource/csv_symptoms_resnet/' + file_name + 'csv', "r"))
    reader = csv.reader(open('resource/csv_symptoms_resnet_best/' + file_name + 'csv', "r"))
    next(reader)
    row = next(reader)
    return np.array([element.rstrip(",") for element in row[column].strip("[]").split()]).astype(np.float)


def vgg_face_compare_symptoms_average_faces(csv_path):
    header = ['name1', 'name2', 'pair index', 'euclidean_distance']

    for t_f_pairs in ('resource/TruePairs', 'resource/FalsePairs'):

        with open(csv_path + t_f_pairs[9:] + '.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            pairs = os.listdir(t_f_pairs)
            for pair in pairs:
                images = glob.glob(t_f_pairs + '/' + pair + '/*.png')
                query_weight1 = read_average_symptoms(os.path.basename(images[0])[:-3], 2)
                query_weight2 = read_average_symptoms(os.path.basename(images[1])[:-3], 2)
                euclidean_distance = np.linalg.norm(query_weight2 - query_weight1, axis=0)
                row = {'name1': os.path.basename(images[0])[:-4],
                       'name2': os.path.basename(images[1])[:-4],
                       'pair index': pair,
                       'euclidean_distance': euclidean_distance
                       }

                writer.writerow(row)


def read_random_symptoms(file_name, column):
    csv.field_size_limit(2000000)

    reader = csv.reader(open('resource/csv_symptoms_resnet/' + file_name + 'csv', "r"))
    rows = list(reader)
    row = rows[random.randint(2, len(rows) - 1)]
    return np.array([element.rstrip(",") for element in row[column].strip("[]").split()]).astype(np.float)


def vgg_face_compare_symptoms_random_frame():
    csv_path = 'resource/euclidean_distances/resnet/random_faces-'

    for t_f_pairs in ('resource/TruePairs', 'resource/FalsePairs'):

        with open(csv_path + t_f_pairs[9:] + '.csv', 'w', encoding='UTF8', newline='') as f:
            header = ['name1', 'name2', 'pair index', 'euclidean_distance']
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            pairs = os.listdir(t_f_pairs)
            for pair in pairs:
                images = glob.glob(t_f_pairs + '/' + pair + '/*.png')
                query_weight1 = read_random_symptoms(os.path.basename(images[0])[:-3], 2)
                query_weight2 = read_random_symptoms(os.path.basename(images[1])[:-3], 2)
                euclidean_distance = np.linalg.norm(query_weight2 - query_weight1, axis=0)
                row = {'name1': os.path.basename(images[0])[:-4],
                       'name2': os.path.basename(images[1])[:-4],
                       'pair index': pair,
                       'euclidean_distance': euclidean_distance
                       }

                writer.writerow(row)


def read_all_frames_symptoms(file_name, column):
    csv.field_size_limit(2000000)

    reader = csv.reader(open('resource/csv_symptoms_resnet/' + file_name + '.csv', "r"))
    next(reader)
    next(reader)
    symptoms = []
    for row in reader:
        symptom = np.array([element.rstrip(",") for element in row[column].strip("[]").split()]).astype(np.float)
        symptoms.append(symptom)
    return symptoms


def vgg_face_compare_symptoms_all_frames_average():
    csv_path = 'resource/euclidean_distances/resnet/all_frames(average)-'

    for t_f_pairs in ('resource/TruePairs', 'resource/FalsePairs'):

        with open(csv_path + t_f_pairs[9:] + '.csv', 'w', encoding='UTF8', newline='') as f:
            header = ['name1', 'name2', 'pair index', 'euclidean_distance']
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            pairs = os.listdir(t_f_pairs)
            for pair in tqdm(pairs):
                persons = os.listdir(t_f_pairs + '/' + pair)
                persons = [x for x in persons if ".png" not in x]

                person1 = read_all_frames_symptoms(persons[0], 2)
                euclidean_distance = 0
                for query_weight1 in person1:
                    person2 = read_all_frames_symptoms(persons[1], 2)
                    for query_weight2 in person2:
                        euclidean_distance += np.linalg.norm(query_weight2 - query_weight1, axis=0) / (
                                len(person1) * len(person2))

                row = {'name1': persons[0],
                       'name2': persons[1],
                       'pair index': pair,
                       'euclidean_distance': euclidean_distance
                       }

                writer.writerow(row)
