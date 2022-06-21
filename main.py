import os
import pandas as pd
import numpy as np
import cv2 as cv
import glob

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from pca import *
from evaluation import *
from vgg_face import *


def cut_face(frame, eyes):
    eye_left_center = [(eyes[36][0] + eyes[40][0]) / 2, (eyes[36][1] + eyes[40][1]) / 2]
    eye_right_center = [(eyes[43][0] + eyes[46][0]) / 2, (eyes[43][1] + eyes[46][1]) / 2]
    dy = eye_right_center[1] - eye_left_center[1]
    dx = eye_right_center[0] - eye_left_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    desiredRightEyeX = 1.0 - 0.35
    dist = np.sqrt((dx ** 2) + (dy ** 2))
    desiredDist = (desiredRightEyeX - 0.35)
    desiredDist *= 100
    scale = desiredDist / dist

    eyesCenter = ((eye_left_center[0] + eye_right_center[0]) // 2,
                  (eye_left_center[1] + eye_right_center[1]) // 2)
    M = cv.getRotationMatrix2D(eyesCenter, angle, scale)

    tX = 100 * 0.5
    tY = 100 * 0.35
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    (w, h) = (100, 100)
    output = cv.warpAffine(frame, M, (w, h), flags=cv.INTER_CUBIC)
    return output


def load_video_save_faces(video_path, video, landmarks2d):
    frames = video.shape[3]

    if not os.path.isdir(video_path):
        os.mkdir(video_path)

    for i in range(frames):
        frame = cv.cvtColor(np.ascontiguousarray(video[:, :, :, i], dtype=np.uint8), cv.COLOR_RGB2BGR)
        frame = cut_face(frame, landmarks2d[:, :, i])
        cv.imwrite(video_path + '/' + str(i) + '.png', frame)


def load_data(video_name, pair_path):
    video_path = pair_path + "/" + video_name[:-4]
    data = np.load('resource/videa/' + video_name)
    video = data['colorImages']
    landmarks2D = data['landmarks2D']
    load_video_save_faces(video_path, video, landmarks2D)


def make_average_face(pair_path, name, all_faces):
    images = glob.glob(pair_path + '/' + name + '/*.png')
    image_data = []
    for img in images:
        this_image = cv.imread(img, 1)
        image_data.append(this_image)
        all_faces.append(this_image)

    avg_image = image_data[0]
    for i in range(len(image_data)):
        if i == 0:
            pass
        else:
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            avg_image = cv.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)

    all_faces.append(avg_image)

    cv.imwrite(pair_path + '/' + name + '.png', avg_image)


def read_pairs(path, all_faces):
    file = pd.read_csv(path)

    if not os.path.isdir(path[:-4]):
        os.mkdir(path[:-4])
    for pair_id in range(len(file.values)):
        pair_path = path[:-4] + "/" + str(pair_id)
        if not os.path.isdir(pair_path):
            os.mkdir(pair_path)
        pair = file.values[pair_id]
        load_data(pair[0], pair_path)
        make_average_face(pair_path, pair[0][:-4], all_faces)
        load_data(pair[1], pair_path)
        make_average_face(pair_path, pair[1][:-4], all_faces)


def show_faces(all_faces):
    # Show sample faces using matplotlib
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
    for i in range(16):
        image = cv.cvtColor(np.ascontiguousarray(all_faces[i * 100], dtype=np.uint8), cv.COLOR_RGB2BGR)
        axes[i % 4][i // 4].imshow(image, cmap="gray")
    plt.show()


def read_faces_from_dic(all_faces):
    for t_f_pairs in ('resource/TruePairs', 'resource/FalsePairs'):
        pairs = os.listdir(t_f_pairs)
        i = 0
        for pair in pairs:
            i += 1
            if i > 30:
                break
            images = glob.glob(t_f_pairs + '/' + pair + '/*.png')
            for img in images:
                this_image = cv.imread(img, 1)
                all_faces.append(this_image)
                # all_faces.append(cv.cvtColor(this_image, cv.COLOR_BGR2GRAY))

            persons = os.listdir(t_f_pairs + '/' + pair)
            for person in persons:
                images = glob.glob(t_f_pairs + '/' + pair + '/' + person + '/*.png')
                for img in images:
                    this_image = cv.imread(img, 1)
                    all_faces.append(this_image)
                    # all_faces.append(cv.cvtColor(this_image, cv.COLOR_BGR2GRAY))


def load_persons():
    path = 'resource/csv_symptoms_resnet_best'
    persons = os.listdir(path)
    symptoms = []
    names = []
    for person in persons:
        reader = csv.reader(open(path + '/' + person, "r"))
        next(reader)
        data = next(reader)

        symptoms.append(np.array([element.rstrip(",") for element in data[2].strip("[]").split()]).astype(np.float))
        names.append(data[0])

    return names, symptoms


def dimension_reduction():
    names, symptoms = load_persons()
    pca = PCA(n_components=3)
    _ = pca.fit(symptoms)
    symptoms = pca.transform(symptoms)

    header = ['name', 'sympthom1', 'sympthom2', 'sympthom3']

    with open("resource/csv_symptoms_resnet_best/symptoms_3d.csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for i in range(len(symptoms)):
            row = {'name': names[i],
                   'sympthom1': symptoms[i][0],
                   'sympthom2': symptoms[i][1],
                   'sympthom3': symptoms[i][2]
                   }
            writer.writerow(row)


def get_symptoms_3d():
    csv.field_size_limit(10000000)
    with open("resource/csv_symptoms_resnet_best/symptoms_3d.csv", 'r') as video_csv:
        reader = csv.reader(video_csv)
        next(reader)
        vectors = []
        names = []
        for row in tqdm(reader):
            vectors.append([float(row[1]), float(row[2]), float(row[3])])
            names.append(row[0])
    return numpy.array(vectors), numpy.array(names)


def cluster_images():
    # dimension_reduction()
    X, names = get_symptoms_3d()
    kmeans = KMeans(n_clusters=4)
    kmeans = kmeans.fit(X)
    labels = kmeans.predict(X)

    x = numpy.array(labels == 0)
    y = numpy.array(labels == 1)
    z = numpy.array(labels == 2)
    w = numpy.array(labels == 3)
    print(names[x])
    print(names[y])
    print(names[z])
    print(names[w])

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[x][:, 0], X[x][:, 1], X[x][:, 2], color='red')
    ax.scatter(X[y][:, 0], X[y][:, 1], X[y][:, 2], color='b')
    ax.scatter(X[z][:, 0], X[z][:, 1], X[z][:, 2], color='yellow')
    ax.scatter(X[w][:, 0], X[w][:, 1], X[w][:, 2], color='m')
    plt.show()


def main():
    all_faces = []
    # read_pairs('resource/TruePairs.csv', all_faces)
    # read_pairs('resource/FalsePairs.csv', all_faces)
    read_faces_from_dic(all_faces)
    show_faces(all_faces)

    main_pca(all_faces)
    # best_threshold = show_rocs('resource/euclidean_distances/pca/')

    # main_vgg_face()
    # best_threshold = show_rocs('resource/euclidean_distances/resnet/')

    best_threshold = show_best_roc('resource/euclidean_distances/best2/')
    print(best_threshold)
    calculate_confusion_matrix('resource/euclidean_distances/best2/', best_threshold)
    # cluster_images()


if __name__ == '__main__':
    main()
