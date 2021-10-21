import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import NearestCentroid
# from sklearn.svm import SVC
# from sklearn.model_selection import KFold, cross_val_score, validation_curve
# from sklearn.ensemble import VotingClassifier, BaggingClassifier
# from sklearn.cluster import KMeans
# import torch as T
# from torch.nn import Linear
# from torch.optim import Adam
# from torch.nn import CrossEntropyLoss
# import torch.nn.functional as F

#plot_clf apo sklearn
#K-fold --> model selection kai cross val score
#TI EINAI TO FOLNT

X_training = np.genfromtxt('C:/Users/miefk/desktop/data/train/train.txt')
X_test = np.genfromtxt('C:/Users/miefk/desktop/data/test/test.txt')
train_features = X_training[:, 1:]
train_labels = X_training[:, 0]
test_features = X_test[:, 1:]
test_labels = X_test[:, 0]


def show_sample(X, index):
    '''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    '''
    picture = X[index, 1:].reshape(16, 16)
    plt.imshow(picture, cmap='binary')
    plt.show()
    raise NotImplementedError


def plot_digits_samples(X, y):
    '''Takes a dataset and selects one example from each label and plots it in subplots
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    '''
    fig, axs = plt.subplots(1, 10)
    digits = set(y)
    t = len(digits)
    ax = 0
    while t != 0:
        for i in range(len(X)):
            if y[i] in digits:
                axs[ax].imshow(X[i].reshape(16, 16), cmap='binary')
                digits.remove(y[i])
                ax += 1
                t -= 1
    plt.show()
    raise NotImplementedError


def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the mean for all instances of a specific digit at a pixel location
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select.
    Returns:
        (float): The mean value of the digits for the specified pixels
    '''
    mean = 0
    for i in range(len(X)):
        mean += X[i, 1:].reshape(16, 16)[pixel[0], pixel[1]]
    mean /= len(X)
    # print(mean)
    return mean


def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    mean = []
    numbers = np.insert(X, 0, y, axis=1)
    numbers = numbers[numbers[:, 0] == digit]
    for i in range(16):
        for j in range(16):
            mean.append(digit_mean_at_pixel(numbers, y, digit, pixel=(i, j)))
    # print(mean)
    # new_pic = np.array(mean).reshape(16, 16)
    # plt.imshow(new_pic, cmap='binary')
    # plt.show()
    return mean
    raise NotImplementedError


def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the variance for all instances of a specific digit at a pixel location
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select
    Returns:
        (float): The variance value of the digits for the specified pixels
    '''
    mean_value = digit_mean_at_pixel(X, y, digit, pixel)
    variance = 0
    for i in range(len(X)):
        if y[i] == digit:
            variance += (X[i, 1:].reshape(16, 16)[pixel[0], pixel[1]] - mean_value) ** 2
    variance /= len(X)
    # print(variance)
    return variance
    raise NotImplementedError


def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    variance = []
    numbers = np.insert(X, 0, y, axis=1)
    numbers = numbers[numbers[:, 0] == digit]
    for i in range(16):
        for j in range(16):
            variance.append(digit_variance_at_pixel(numbers, y, digit, pixel=(i, j)))
    # print(variance)
    return variance
    raise NotImplementedError


def euclidean_distance(s, m):
    '''Calculates the euclidean distance between a sample s and a mean template m
    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)
    Returns:
        (float) The Euclidean distance between s and m
    '''
    return np.linalg.norm(s - m)
    raise NotImplementedError


def euclidean_distance_classifier(X, X_mean):
    '''Classifiece based on the euclidean distance between samples in X and template vectors in X_mean
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)
    Returns:
        (np.ndarray) predictions (nsamples)
    '''
    predictions = []
    total = len(X)
    hit = 0
    for i in range(len(X)):
        distances = []
        for j in range(10):
            distances.append(euclidean_distance(X[i], X_mean[j]))
        min_distance = min(distances)
        prediction = distances.index(min_distance)
        predictions.append(prediction)
        if test_labels[i] == prediction:
            hit += 1
    accuracy = 100 * hit / total
    return predictions, accuracy


# mean_X = [digit_mean(train_features, train_labels, i) for i in range(10)]
# euclidean_distance_classifier(test_features, mean_X)


class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        Calculates self.X_mean_ based on the mean
        feature values in X for each class.
        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)
        fit always returns self.
        """
        self.X_mean_ = [digit_mean(X, y, i) for i in range(10)]
        # raise NotImplementedError
        return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        results = euclidean_distance_classifier(X, self.X_mean_)
        predictment = results[0]
        print('The predictions for the test samples given by the model are: ', predictment)
        return predictment
        # raise NotImplementedError


    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        results = euclidean_distance_classifier(X, self.X_mean_)
        acc = results[1]
        print('The accuracy of the hand-made model is: ', acc, '%')
        return acc
        raise NotImplementedError

# c = EuclideanDistanceClassifier()
# c.fit(train_features, train_labels)
# c.predict(test_features)
# c.score(test_features, test_labels)




# def evaluate_sklearn_nb_classifier(X, y, folds=5):
#     """ Create an sklearn naive bayes classifier and evaluate it using cross-validation
#     Calls evaluate_classifier
#     """
#     clf = GaussianNB()
#     clf.fit(train_features, train_labels)
#     # prediction = clf.predict(test_features)
#     cross_val = cross_val_score(clf, X, y)
#     print(cross_val)
#     return cross_val
#     raise NotImplementedError


# def evaluate_knn_classifier(X, y, folds=5):
#     """ Create a knn and evaluate it using cross-validation
#     Calls evaluate_classifier
#     """
#     folding = KFold(folds)
#     clf = NearestCentroid()
#     clf.fit(train_features, train_labels)
#     cross_val = 100 * cross_val_score(clf, X, y)
#     cross_val_score()
#     score = np.mean(cross_val)
#     print(score)
#     raise NotImplementedError
#
#
# def evaluate_linear_svm_classifier(X, y, folds=5):
#     """ Create an svm with linear kernel and evaluate it using cross-validation
#     Calls evaluate_classifier
#     """
#     clf = SVC(kernel='linear')
#     clf.fit(train_features, train_labels)
#     cross_val = cross_val_score(clf, X, y)
#     score = np.mean(cross_val)
#     print(score)
#     raise NotImplementedError
#
#
# def evaluate_rbf_svm_classifier(X, y, folds=5):
#     """ Create an svm with rbf kernel and evaluate it using cross-validation
#     Calls evaluate_classifier
#     """
#     clf = SVC(kernel='rbf')
#     clf.fit(train_features, train_labels)
#     cross_val = cross_val_score(clf, X, y)
#     score = np.mean(cross_val)
#     print(score)
#     raise NotImplementedError
#
#
# def evaluate_nn_classifier(X, y, folds=5):
#     """ Create a pytorch nn classifier and evaluate it using cross-validation
#     Calls evaluate_classifier
#     """
#     input_data = T.tensor(train_features, dtype=T.float32)
#     fc1 = Linear(256, 100)
#     fc2 = Linear(100, 100)
#     fc3 = Linear(100, 10)
#
#     def forward():
#         x = fc1(input_data)
#         x = F.relu(x)
#         x = fc2(x)
#         x = F.relu(x)
#         x = fc3(x)
#         x = F.relu(x)
#
#     # print(out)
#
#
#
#
#     raise NotImplementedError
#
#
# def evaluate_voting_classifier(X, y, folds=5):
#     """ Create a voting ensemble classifier and evaluate it using cross-validation
#     Calls evaluate_classifier
#     """
#     clf = VotingClassifier()
#     clf.fit(train_features, train_labels)
#     cross_val = cross_val_score(clf, X, y)
#     score = np.mean(cross_val)
#     print(score)
#     raise NotImplementedError
#
#
# def evaluate_bagging_classifier(X, y, folds=5):
#     """ Create a bagging ensemble classifier and evaluate it using cross-validation
#     Calls evaluate_classifier
#     """
#     clf = BaggingClassifier()
#     clf.fit(train_features, train_labels)
#     cross_val = cross_val_score(clf, X, y)
#     score = np.mean(cross_val)
#     print(score)
#     raise NotImplementedError

# evaluate_sklearn_nb_classifier(test_features, test_labels)
# evaluate_knn_classifier(test_features, test_labels)
# evaluate_linear_svm_classifier(test_features, test_labels)
# evaluate_rbf_svm_classifier(test_features, test_labels)
# evaluate_nn_classifier(test_features, test_labels)
# evaluate_voting_classifier(test_features, test_labels)  θέλει extra argument sto clf
# evaluate_bagging_classifier(test_features, test_labels)