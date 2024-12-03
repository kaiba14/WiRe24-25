import numpy as np
import lib
import matplotlib as mpl


####################################################################################################
# Exercise 1: Power Iteration

def power_iteration(M: np.ndarray, epsilon: float = -1.0) -> (np.ndarray, list):
    """
    Compute largest eigenvector of matrix M using power iteration. It is assumed that the
    largest eigenvalue of M, in magnitude, is well separated.

    Arguments:
    M: matrix, assumed to have a well separated largest eigenvalue
    epsilon: epsilon used for convergence (default: 10 * machine precision)

    Return:
    vector: eigenvector associated with largest eigenvalue
    residuals: residual for each iteration step

    Raised Exceptions:
    ValueError: if matrix is not square

    Forbidden:
    numpy.linalg.eig, numpy.linalg.eigh, numpy.linalg.svd
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix not nxn")

    # TODO: set epsilon to default value if not set by user
    if epsilon < 0:
        epsilon = 1e-10
    # TODO: normalized random vector of proper size to initialize iteration
    n = M.shape[0]
    vector = np.random.rand(n)
    vector /= np.linalg.norm(vector)
    # Initialize residual list and residual of current eigenvector estimate
    residuals = []
    residual = 2.0 * epsilon

    # Perform power iteration
    while residual > epsilon:
        vector_new = M.dot(vector)

        eigenvalue = np.dot(vector, vector_new)

        vector_new /= np.linalg.norm(vector_new)

        residual = np.linalg.norm(vector_new - vector)

        vector = vector_new

        residuals.append(residual)

    return vector, residuals


####################################################################################################
# Exercise 2: Eigenfaces

def load_images(path: str, file_ending: str=".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """
    images = []

    files = lib.list_directory(path)

    image_files = [file for file in files if file.endswith(file_ending)]

    image_files.sort(key=lambda x: [int(part) if part.isdigit() else part for part in x.split('_')])

    for file in image_files:
        image_path = f"{path}/{file}"
        image = mpl.image.imread(image_path)
        images.append(np.asarray(image, dtype=np.float64))

    if images:
        dimension_y, dimension_x = images[0].shape
    else:
        dimension_y, dimension_x = 0, 0

    return images, dimension_x, dimension_y

def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    Arguments:
    images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    Return:
    D: data matrix that contains the flattened images as rows
    """
    # TODO: initialize data matrix with proper size and data type
    D = np.zeros((0, 0))
    num_images = len(images)
    if num_images == 0:
        raise ValueError("No images provided in the list.")

    image_shape = images[0].shape
    num_pixels = np.prod(image_shape)
    D = np.zeros((num_images, num_pixels), dtype=np.float64)

    # TODO: add flattened images to data matrix
    for i, image in enumerate(images):
        D[i, :] = image.flatten()

    return D

def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform principal component analysis for given data matrix.

    Arguments:
    D: data matrix of size m x n where m is the number of observations and n the number of variables

    Return:
    pcs: matrix containing principal components as rows
    svals: singular values associated with principle components
    mean_data: mean that was subtracted from data
    """

    if D.shape[0] == 0:
        raise ValueError("Data matrix is empty.")

    mean_data = np.mean(D, axis=0)

    centered_data = D - mean_data

    _, svals, pcs = np.linalg.svd(centered_data, full_matrices=False)

    return pcs, svals, mean_data

def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    Arguments:
    singular_values: vector containing singular values
    threshold: threshold for determining k (default = 0.8)

    Return:
    k: threshold index
    """

    normalized_svals = singular_values / np.sum(singular_values)

    k = 0
    cumulative_energy = 0.0
    total_energy = threshold * np.sum(singular_values)

    for i, sval in enumerate(normalized_svals):
        cumulative_energy += sval
        if cumulative_energy >= threshold:
            k = i + 1
            break

    return k

def project_faces(pcs: np.ndarray, images: list, mean_data: np.ndarray) -> np.ndarray:
    """
    Project given image set into basis.

    Arguments:
    pcs: matrix containing principal components / eigenfunctions as rows
    images: original input images from which pcs were created
    mean_data: mean data that was subtracted before computation of SVD/PCA

    Return:
    coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """

    # TODO: initialize coefficients array with proper size
    coefficients = np.zeros((1, 1))
    num_images = len(images)
    num_pcs = pcs.shape[0]
    coefficients = np.zeros((num_images, num_pcs))
    
    # TODO: iterate over images and project each normalized image into principal component basis
    for i, image in enumerate(images):
        flattened_image = image.flatten() - mean_data
        image_coefficients = np.dot(flattened_image, pcs.T)
        coefficients[i, :] = image_coefficients

    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (
np.ndarray, list, np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data

    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """


    # TODO: load test data set
    imgs_test, _, _ = load_images(path_test)

    # TODO: project test data set into eigenbasis
    coeffs_test = project_faces(pcs, imgs_test, mean_data)

    # TODO: Initialize scores matrix with proper size
    scores = np.zeros((coeffs_train.shape[0], coeffs_test.shape[0]))

    # TODO: Iterate over all images and calculate pairwise correlation
    for i in range(coeffs_train.shape[0]):
        for j in range(coeffs_test.shape[0]):
            angle = np.arccos(np.dot(coeffs_train[i], coeffs_test[j]) /
                             (np.linalg.norm(coeffs_train[i]) * np.linalg.norm(coeffs_test[j])))
            scores[i, j] = angle

    return scores, imgs_test, coeffs_test



if __name__ == '__main__':

    A = np.random.randn( 7, 7)
    A = A.transpose().dot(A)
    L,U = np.linalg.eig( A)
    L[1] = L[0] - 10**-3
    A = U.dot(np.diag(L)).dot(U.transpose())
    print( )
    np.set_printoptions(precision=16)
    print( A.flatten())

    A = np.array( [ 18.2112344794043359,   0.7559886314903312,  7.2437569750169502,
                    -13.8991061752623271,   4.8768689715057691,  -1.318055436971276,
                    -6.7829844205260148,   0.7559886314903312,   7.9204801042364448,
                     1.5378938590357767,   7.1775560914639325,   2.8536549530686015,
                     1.9998683983340397,  -5.9532930598376685,   7.2437569750169502,
                     1.5378938590357767,   9.841906218619128,   0.5841092845624152,
                     6.7510103134860797,   4.6111951240722888,  -8.9825300821798191,
                    -13.8991061752623271,   7.1775560914639334,   0.5841092845624152,
                     24.2028041177043818,   0.8180957104689988,   6.6087248591945729,
                    -4.1573996873552073,   4.8768689715057691,   2.8536549530686015,
                     6.7510103134860806,   0.8180957104689979,   7.0366782892027206,
                     5.4944303652858073,  -9.0773671527609796,  -1.318055436971276,
                     1.9998683983340397,   4.6111951240722888,   6.608724859194572,
                     5.4944303652858073,   8.1889694453300805,  -7.1176432086570651,
                    -6.7829844205260148,  -5.9532930598376685,  -8.9825300821798191,
                    -4.1573996873552046,  -9.0773671527609796,  -7.1176432086570633,
                    13.664209790087753 ])
    A = A.reshape( (7,7))

    ev, res = power_iteration( A)



    print( 'ev = ' + str(ev))

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")