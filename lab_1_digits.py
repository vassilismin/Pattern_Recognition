import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold, cross_val_score, learning_curve
# from sklearn.neighbors import NearestCentroid

'''This python file implements the Euclidean distance classifier to solve the 
hand-written digit classification problem.'''


'''The first step is to load the train data and test data and extract the features and labels. 
This is done using the numpy library.'''
X_training = np.genfromtxt('C:/Users/miefk/desktop/data/train/train.txt')
X_test = np.genfromtxt('C:/Users/miefk/desktop/data/test/test.txt')
train_features = X_training[:, 1:]
train_labels = X_training[:, 0]
test_features = X_test[:, 1:]
test_labels = X_test[:, 0]


'''The second step is to plot (some of) the samples to get a better idea of what they look like. 
This can be done with the matplotlib library using imshow.'''


def show_sample(X, index):
    """Plots any one of the samples we want. We use the binary colormap, which converts the picture to grey-scale
        The inputs are the training features (np.array) and the desired index (int)"""
    picture = X[index, :].reshape(16, 16)
    plt.imshow(picture, cmap='binary')
    plt.show()

# show_sample(train_features, 131)


'''The third step is to plot one example from each label, so we expect to see all the digits from 0 to 9'''


def plot_digits_samples(X, y):
    """Plots all digits from 0 to 9 using a grey-scale colormap. The inputs are the training features (np.array)
        and the training labels (np.array)"""
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


# plot_digits_samples(train_features, train_labels)


'''The fourth step is to compute the mean value of the pixel (10, 10) for a given digit of the training features'''


def mean_value_at_pixel_of_digit(X, y, digit, pixel=(10, 10)):
    """Computes the mean value of a specific pixel for all instances of a digit in the training set. The inputs
    are the training features (np.array), the training labels (np.array), the desired digit (int) and the desired
    pixel (tuple). The default pixel is (10, 10)"""
    mean = 0
    for i in range(len(X)):
        if y[i] == digit:
            mean += X[i, :].reshape(16, 16)[pixel[0], pixel[1]]
    mean /= len(X)
    print(mean)
    return mean

# mean_value_at_pixel_of_digit(train_features, train_labels, 0)


'''The fifth step is to do the same computation as the fourth step but this time for the variance of a pixel'''


def variance_at_pixel_of_digit(X, y, digit, pixel=(10, 10)):
    """Computes the variance of a specific pixel for all instances of a digit in the training set. The inputs
    are the training features (np.array), the training labels (np.array), the desired digit (int) and the desired
    pixel (tuple). The default pixel is (10, 10)"""
    mean_value = mean_value_at_pixel_of_digit(X, y, digit, pixel)
    variance = 0
    for i in range(len(X)):
        if y[i] == digit:
            variance += (X[i, :].reshape(16, 16)[pixel[0], pixel[1]] - mean_value) ** 2
    variance /= len(X)
    print(variance)
    return variance

# variance_at_pixel_of_digit(train_features, train_labels, 0)


'''The sixth step is to compute the mean value and the variance of every pixel of all instances of a given digit'''

'''We begin with the mean value. We use a slightly different implementation. We do not call mean_value_at_pixel_of_digit
function. We implement the following function instead'''
def set_mean_at_pixel(X, pixel=(10, 10)):
    """This function is slightly different than mean_value_at_pixel_of_digit. It computes the mean value of a
     given set. This works in collaboration with the next function, digit_mean"""
    mean = 0
    for i in range(len(X)):
        mean += X[i, 1:].reshape(16, 16)[pixel[0], pixel[1]]
    mean /= len(X)
    # print(mean)
    return mean

def digit_mean(X, y, digit):
    """This function uses the set_mean_at_pixel to compute the mean_value of all instances of a digit. We specify the
    desired digit (int), the training features (X: np.array) and the training labels (y: np.array)"""
    mean = []
    numbers = np.insert(X, 0, y, axis=1)
    numbers = numbers[numbers[:, 0] == digit]
    for i in range(16):
        for j in range(16):
            mean.append(set_mean_at_pixel(numbers, pixel=(i, j)))
    """If we want to plot the desired digit according to the mean values we have to un-comment the following lines"""
    # new_pic = np.array(mean).reshape(16, 16)
    # plt.imshow(new_pic, cmap='binary')
    # plt.show()
    return mean


# digit_mean(train_features, train_labels, 0)


'''Next is the variance'''

def set_variance_at_pixel(X, pixel=(10, 10)):
    """This function is slightly different than variance_at_pixel_of_digit. It computes the variance of a
     given set X (np.array). This works in collaboration with the next function, digit_variance"""
    mean_value = set_mean_at_pixel(X, pixel)
    variance = 0
    for i in range(len(X)):
            variance += (X[i, 1:].reshape(16, 16)[pixel[0], pixel[1]] - mean_value) ** 2
    variance /= len(X)
    return variance

def digit_variance(X, y, digit):
    """This function uses the set_mean_at_pixel to compute the mean_value of all instances of a digit. We specify the
    desired digit (int), the training features (X: np.array) and the training labels (y: np.array)"""
    variance = []
    numbers = np.insert(X, 0, y, axis=1)
    numbers = numbers[numbers[:, 0] == digit]
    for i in range(16):
        for j in range(16):
            variance.append(set_variance_at_pixel(numbers, pixel=(i, j)))
    """If we want to plot the desired digit according to the variance we have to un-comment the following lines"""
    # new_pic = np.array(variance).reshape(16, 16)
    # plt.imshow(new_pic, cmap='binary')
    # plt.show()
    return variance

# digit_variance(train_features, train_labels, 0)



'''The seventh step is to plot a given digit using the mean values of the all the pixels of all instances of 
the digit. We can use the digit_mean function above but we have to un-comment the last three lines'''

def plot_digit_mean(X, y, digit):
    """This function uses the digit_mean_at_pixel to compute the mean_value of all instances of a digit. We specify the
    desired digit (int), the training features (X: np.array) and the training labels (y: np.array)"""
    mean = []
    numbers = np.insert(X, 0, y, axis=1)
    numbers = numbers[numbers[:, 0] == digit]
    for i in range(16):
        for j in range(16):
            mean.append(set_mean_at_pixel(numbers, pixel=(i, j)))
    """If we want to plot the desired digit according to the mean values we have to un-comment the following lines"""
    new_pic = np.array(mean).reshape(16, 16)
    plt.imshow(new_pic, cmap='binary')
    plt.show()
    return mean

# plot_digit_mean(train_features, train_labels, 0)


'''The eighth step is to plot a given digit using the variance of the all the pixels of all instances of 
the digit. We can use the digit_variance function above but we have to un-comment the last three lines'''


def plot_digit_variance(X, y, digit):
    """This function uses the set_mean_at_pixel to compute the mean_value of all instances of a digit. We specify the
    desired digit (int), the training features (X: np.array) and the training labels (y: np.array)"""
    variance = []
    numbers = np.insert(X, 0, y, axis=1)
    numbers = numbers[numbers[:, 0] == digit]
    for i in range(16):
        for j in range(16):
            variance.append(set_variance_at_pixel(numbers, pixel=(i, j)))
    """If we want to plot the desired digit according to the variance we have to un-comment the following lines"""
    new_pic = np.array(variance).reshape(16, 16)
    plt.imshow(new_pic, cmap='binary')
    plt.show()


# plot_digit_variance(train_features, train_labels, 0)



'''The ninth step is to compute the mean value and variance of all pixels of all digits and plot the accordingly.
This can be done by using list-comprehension and the functions of step 7 and 8.'''

'''The array of the mean values of all digits is given by: '''
# mean_X = [digit_mean(train_features, train_labels, i) for i in range(10)]

'''The array of the variances of all digits is given by: '''
# var_X = [digit_variance(train_features, train_labels, i) for i in range(10)]


'''To plot the digits we can use the following two function: '''

def plot_digits_by_mean(X, y):
    """Iterates through all numbers from 0 to 9, plots them according to the mean value and puts them all
     on the same figure"""
    fig, axs = plt.subplots(1, 10)
    for i in range(10):
        axs[i].imshow(np.array(digit_mean(X, y, i)).reshape(16, 16), cmap='binary')
    plt.show()

# plot_digits_by_mean(train_features, train_labels)


def plot_digits_by_variance(X, y):
    """Iterates through all numbers from 0 to 9, plots them according to the variance and puts them all
     on the same figure"""
    fig, axs = plt.subplots(1, 10)
    for i in range(10):
        axs[i].imshow(np.array(digit_variance(X, y, i)).reshape(16, 16), cmap='binary')
    plt.show()


# plot_digits_by_variance(train_features, train_labels)



'''Now we can start classifying the test samples according to the euclidean distance criterion. We will need a 
function that computes this distance between two arrays.'''

def euclidean_distance(s, m):
    """Calculates the euclidean distance between a sample s and a mean template m
    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)
    Returns:
        (float) The Euclidean distance between s and m
    """
    return np.linalg.norm(s - m)


'''Next, we need a function that implements the euclidean distance classifier.'''

def euclidean_distance_classifier(X, X_mean):
    """This function classifies all instances of a given set according to the euclidean distance from the (given)
     mean value of the training set. For each instance we compute the euclidean distance of the instance from the mean
     value (centroid) of each class and find the minimum. This minimum distance points towards the class to which the
     instance is classified"""

    """The first  case is when the input is one single instance. 
        The second case is when the input is multiple instances"""
    if X.shape[0] == 256:
        distances = []
        for j in range(10):
            distances.append(euclidean_distance(X, X_mean[j]))
        min_distance = min(distances)
        prediction = distances.index(min_distance)
        return prediction
    else:
        predictions = []
        total = len(X)
        for i in range(len(X)):
            distances = []
            for j in range(10):
                distances.append(euclidean_distance(X[i], X_mean[j]))
            min_distance = min(distances)
            prediction = distances.index(min_distance)
            predictions.append(prediction)
        accuracy = np.sum(test_labels == predictions) / total
        return predictions, accuracy


'''The tenth step is to classify the test feature that has index 101'''
# mean_X = [digit_mean(train_features, train_labels, i) for i in range(10)]
# pred = euclidean_distance_classifier(test_features[101], mean_X)
# target = test_labels[101]
# if pred == target:
#     print('The classification was correct!')
# else:
#     print('The classification was not correct. The prediction was:', pred, 'but the correct label is:', target)


'''We can see that the classifier did not work in this case.'''


'''The eleventh step is to classify all instances of the test set, measure the accuracy and store the predictions
in an array'''
# mean_X = [digit_mean(train_features, train_labels, i) for i in range(10)]
# preds, acc = euclidean_distance_classifier(test_features, mean_X)
# print('The accuracy of the euclidean distance classifier is:', 100 * acc, '%')
# print('The array of predictions given by the classifier is: ', '\n', preds)



'''Step number 12 is to make a scikit-learn-like classifier. We implement a class called EuclideanDistanceClassifier 
 and use all the necessary functions created above.'''

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
        return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        preds, _ = euclidean_distance_classifier(X, self.X_mean_)
        return preds


    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        _, acc = euclidean_distance_classifier(X, self.X_mean_)
        return acc

# c = EuclideanDistanceClassifier()
# c.fit(train_features, train_labels)
# # c.predict(test_features)
# c.score(test_features, test_labels)




'''Step 13 is the further investigation and evaluation of our model'''

'''We first compute the cross-validation error using 5-fold validation'''
# def k_fold_cv(X, y, folds=5):
#     """Calls the Euclidean Distance Classifier and computes the cross-validation error using sklearn's class KFOLD.
#     The inputs are the training features X (np.array) and the training labels y (np.array)"""
#     clf = EuclideanDistanceClassifier()
#     cross_val = cross_val_score(clf, X, y, cv=KFold(n_splits=folds), scoring="accuracy")
#     print("CV error = %f +-%f" % (100 * np.mean(cross_val), 100 * np.std(cross_val)))
#
# k_fold_cv(train_features, train_labels)


'''Next we plot the learning curve of the classifier.'''
def plot_learning_curve(X, y):
        clf = EuclideanDistanceClassifier()
        train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=KFold(n_splits=5), scoring='accuracy')
        plt.figure()
        plt.title("Learning Curve")

        plt.ylim(0, 1)
        plt.xlabel("Training examples")
        plt.ylabel("Score")

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        plt.show()


# plot_learning_curve(train_features, train_labels)
