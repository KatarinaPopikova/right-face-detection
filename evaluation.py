import csv
import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc
import seaborn as sns


def plot_roc(tpr, fpr, title, color, ls):
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, ls=ls, label=title + ': %0.2f' % roc_auc)
    plt.legend(loc="lower right")


def start_plot():
    plt.plot([0, 1], [0, 1], color="black", ls="-")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('ROC')


def show_rocs(sub_path):
    plt.figure()
    start_plot()

    tpr, fpr, best_threshold1 = roc_faces(read_col_from_csv(sub_path + 'average_faces-TruePairs.csv', 3),
                                          read_col_from_csv(sub_path + 'average_faces-FalsePairs.csv', 3),
                                          'average faces')
    plot_roc(tpr, fpr, 'average faces', 'y', "-.")

    tpr, fpr, best_threshold2 = roc_faces(read_col_from_csv(sub_path + 'random_faces-TruePairs.csv', 3),
                                          read_col_from_csv(sub_path + 'random_faces-FalsePairs.csv', 3),
                                          'random faces')
    plot_roc(tpr, fpr, 'random faces', 'm', '--')

    tpr, fpr, best_threshold3 = roc_faces(read_col_from_csv(sub_path + 'all_frames(average)-TruePairs.csv', 3),
                                          read_col_from_csv(sub_path + 'all_frames(average)-FalsePairs.csv', 3),
                                          'all(10) frames')
    plot_roc(tpr, fpr, 'all(10) frames', 'c', ':')
    plt.show()
    return max(best_threshold1, best_threshold2, best_threshold3)


def show_best_roc(sub_path):
    plt.figure()
    start_plot()
    tpr, fpr, best_threshold = roc_faces(read_col_from_csv(sub_path + 'average_faces-TruePairs.csv', 3),
                                         read_col_from_csv(sub_path + 'average_faces-FalsePairs.csv', 3),
                                         'average faces')
    plot_roc(tpr, fpr, 'average faces', 'y', "-.")

    plt.show()
    return best_threshold


def read_col_from_csv(file_name, column):
    reader = csv.reader(open(file_name, "r"))
    next(reader)
    return [float(row[column]) for row in reader]


def roc_faces(true_pairs, false_pairs, data_type):
    fpr = []
    tpr = []
    best_threshold = 0

    min_distance = int(min(true_pairs + false_pairs)) - 1
    max_distance = int(max(true_pairs + false_pairs)) + 100
    step = (max_distance - min_distance) / 100

    min_vector_distance = 1

    for threshold in np.arange(min_distance, max_distance, step):
        tpr.append(sum(distance < threshold for distance in true_pairs) / len(true_pairs))
        fpr.append(sum(distance < threshold for distance in false_pairs) / len(false_pairs))

        vector_distance = math.dist([tpr[-1], fpr[-1]], [1, 0])
        if vector_distance < min_vector_distance:
            best_threshold = threshold
            min_vector_distance = vector_distance

    print("Best threshold for " + data_type + ": " + str(best_threshold))
    return tpr, fpr, best_threshold


def evaluate_euclidean(file, threshold):
    next(file)
    over_euclidean = 0
    for row in file:
        if float(row[3]) < threshold:
            over_euclidean += 1
    return over_euclidean, file.line_num - 1 - over_euclidean


def calculate_confusion_matrix(path, threshold):
    tp, fn = evaluate_euclidean(csv.reader(open(path + 'average_faces-TruePairs.csv', "r")), threshold)
    fp, tn = evaluate_euclidean(csv.reader(open(path + 'average_faces-FalsePairs.csv', "r")), threshold)

    ax = sns.heatmap([[tp, fn], [fp, tn]], annot=True, cmap='Blues', fmt='g')
    ax.xaxis.set_ticklabels(['True', 'False'])
    ax.yaxis.set_ticklabels(['True', 'False'])
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Real Values ')

    plt.show()
