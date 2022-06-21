import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from comparator import *


def show_eigenfaces_and_mean_face(eigenfaces, mean_face, faceshape):
    fig, axes = plt.subplots(10, 5, sharex=True, sharey=True, figsize=(8, 10))

    for i in range(50):
        axes[i % 10][i // 10].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")

    plt.show()

    plt.imshow(mean_face, cmap=plt.cm.bone)
    plt.show()


def make_pca(all_faces):
    faceshape = all_faces[0].shape
    facematrix = []
    for image in all_faces:
        facematrix.append(image.flatten())

    facematrix = np.array(facematrix)
    n_components = 50
    pca = PCA(n_components=n_components).fit(facematrix)
    eigenfaces = pca.components_

    # show_eigenfaces_and_mean_face(eigenfaces, pca.mean_.reshape(faceshape), faceshape)

    return eigenfaces, pca.mean_


def main_pca(all_faces):
    eigenfaces, mean_face = make_pca(all_faces)
    # compare_symptoms_average_faces(eigenfaces, mean_face, 'resource/euclidean_distances/pca/average_faces-')
    compare_symptoms_average_faces(eigenfaces, mean_face, 'resource/euclidean_distances/best2/average_faces-')
    # compare_symptoms_random_frame(eigenfaces, mean_face)
    # compare_symptoms_all_frames_average(eigenfaces, mean_face)
