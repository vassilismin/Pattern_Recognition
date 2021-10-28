import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import torch as T
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier
import seaborn as sns
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


def mean_value_of_digit_at_pixel(X, y, digit, pixel=(10,10)):
    index = [ind for ind, x in np.ndenumerate(y) if y[ind] == digit]
    val = [X[ind, :].reshape(16, 16)[pixel] for ind in index]
    return np.mean(val)


'''The fifth step is to do the same computation as the fourth step but this time for the variance of a pixel'''

def variance_of_digit_at_pixel(X, y, digit, pixel=(10, 10)):
    index = [ind for ind, x in np.ndenumerate(y) if y[ind] == digit]
    val = [X[ind, :].reshape(16, 16)[pixel] for ind in index]
    return np.var(val)



'''The sixth step is to compute the mean value and the variance of every pixel of all instances of a given digit'''

'''We begin with the mean value'''
def digit_mean(X, y, digit):
    index = [ind for ind, x in np.ndenumerate(y) if y[ind] == digit]
    val = [X[ind, :] for ind in index]
    return np.mean(val, axis=0)


# print(digit_mean(train_features, train_labels, 0).shape)

'''Next is the variance'''

def digit_variance(X, y, digit):
    index = [ind for ind, x in np.ndenumerate(y) if y[ind] == digit]
    val = [X[ind, :] for ind in index]
    return np.var(val, axis=0)[0]



'''The seventh step is to plot a given digit using the mean values of the all the pixels of all instances of 
the digit. We can use the digit_mean function above but we have to un-comment the last three lines'''

def plot_digit_mean(X, y, digit):
    """This function uses the digit_mean to compute the mean value of all instances of a digit. We specify the
    desired digit (int), the training features (X: np.array) and the training labels (y: np.array)"""
    new_pic = np.array(digit_mean(X, y, digit)).reshape(16, 16)
    plt.imshow(new_pic, cmap='binary')
    plt.show()


# plot_digit_mean(train_features, train_labels, 0)


'''The eighth step is to plot a given digit using the variance of the all the pixels of all instances of 
the digit. We can use the digit_variance function above but we have to un-comment the last three lines'''


def plot_digit_variance(X, y, digit):
    """This function uses the digit_variance to compute the mean_value of all instances of a digit. We specify the
    desired digit (int), the training features (X: np.array) and the training labels (y: np.array)"""
    new_pic = np.array(digit_variance(X, y, digit)).reshape(16, 16)
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


def plot_digits_by_means(X, y):
    """Iterates through all numbers from 0 to 9, plots them according to the mean value and puts them all
     on the same figure"""
    fig, axs = plt.subplots(1, 10)
    for i in range(10):
        axs[i].imshow(np.array(digit_mean(X, y, i)).reshape(16, 16), cmap='binary')

    plt.show()

# plot_digits_by_means(train_features, train_labels)

def plot_digits_by_variance(X, y):
    """Iterates through all numbers from 0 to 9, plots them according to the variance and puts them all
     on the same figure"""
    fig, axs = plt.subplots(1, 10)
    for i in range(10):
        axs[i].imshow(digit_variance(X, y, i).reshape(16, 16), cmap='binary')
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
        predictions = np.array(predictions)
        accuracy = np.sum(test_labels == predictions) / total
        return predictions, accuracy


'''The tenth step is to classify the test feature that has index 101'''
# show_sample(test_features, 101)
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
# print(f'The accuracy of the euclidean distance classifier is: {100 * acc:.2f}', '%')
# print('The array of predictions given by the classifier is: ', '\n', preds)



'''Step number 12 is to make a scikit-learn-like classifier. We implement a class called EuclideanDistanceClassifier 
 and use all the necessary functions created above.'''

class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None

    def fit(self, X, y, principal_component_analysis=False):
        """
        This should fit classifier. All the "work" should be done here.
        Calculates self.X_mean_ based on the mean
        feature values in X for each class.
        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)
        fit always returns self.
        """
        if principal_component_analysis:
            self.X_mean_ = np.array([digit_mean(X, y, i)[0] for i in range(10)])
            trans = PCA(n_components=2)
            self.X_mean_ = trans.fit_transform(self.X_mean_)
        else:
            self.X_mean_ = np.array([digit_mean(X, y, i) for i in range(10)])
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
# print(c.score(test_features, test_labels))




'''Step 13 is the further investigation and evaluation of our model'''

'''We first compute the cross-validation error using 5-fold validation'''
def k_fold_cv(X, y, folds=5):
    """Calls the Euclidean Distance Classifier and computes the cross-validation error using sklearn's class KFOLD.
    The inputs are the training features X (np.array) and the training labels y (np.array)"""
    clf = EuclideanDistanceClassifier()
    cross_val = cross_val_score(clf, X, y, cv=KFold(n_splits=folds), scoring="accuracy")
    print("The 5-fold cross-validation score is %f +-%f" % (100 * np.mean(cross_val), 100 * np.std(cross_val)))
    print("The 5-fold cross-validation error is %f +-%f" % (100 - 100 * np.mean(cross_val), 100 * np.std(cross_val)))
#
#
# k_fold_cv(train_features, train_labels)

'''Next we plot the decision surface of the classifier. Because the dimension of the feature space is very high (256)
we have to implement a dimensionality-reduction algorithm to our data. We use sklearn's Principal Component Analysis
class to reduce the dimension of our data to 2'''
def plot_clf(clf, X, y):
    """We firstly implement the PCA algorithm"""
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(X)
    clf.fit(X_new, y, principal_component_analysis=True)

    """Next we create the canvas of our desired plot"""
    fig, ax = plt.subplots()
    title = ('Decision surface of the Euclidean Distance Classifier')
    X0, X1 = X_new[:, 0], X_new[:, 1]

    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                         np.arange(y_min, y_max, .05))

    """We now have to get the predictions according to our classifier"""
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.9)

    """We plot (scatter the data)"""
    for i in range(10):
        ax.scatter(X0[y == i], X1[y == i], label=i, s=50, alpha=0.9)

    ax.set_ylabel('Principal Component 1')
    ax.set_xlabel('Principal Component 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()


# plot_clf(EuclideanDistanceClassifier(), train_features, train_labels)


'''Next we plot the learning curve of the classifier.'''
def plot_learning_curve(clf, X, y):
        train_sizes, train_scores, test_scores = learning_curve(clf, X, y, scoring='accuracy')
        plt.figure()
        plt.title("Learning Curve of the Neural Network")

        plt.ylim(0.8, 1)
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

# plot_learning_curve(EuclideanDistanceClassifier(), train_features, train_labels)




'''At this point we are done with the Euclidean Distance classifier. Next, we want to implement a scikit-learn-like
 Naive Bayes classifier.'''

'''Step 14 is to calculate the a-priori probabilites for each class (digit)'''

def calculate_priors(y):
    """y (np.array): training labels"""
    """Counts how many times a digits appears inside the labels and returns a dictionary with the probability for
    each digit"""
    probs = {}
    for i in range(10):
        probs[i] = (len(y[y == i]) / len(y))
    # print(probs)
    return probs


# calculate_priors(train_labels)




'''Step 15 is to construct a class that will represent the Naive Bayes classifier'''
class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance
        self.X_mean = None
        self.X_var = None
        self.priors = None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        Calculates self.X_mean_ based on the mean
        feature values in X for each class.
        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)
        fit always returns self.
        """
        self.X_mean = np.array([digit_mean(X, y, i) for i in range(10)])
        self.priors = calculate_priors(y)
        if self.use_unit_variance:
            self.X_var = np.diag([1 for i in range(256)])
        else:
            self.X_var = np.array([digit_variance(X, y, i) for i in range(10)])
            # self.X_var = np.array([vars(X, y, i) for i in range(10)])
        return self


    def gaussian_prob(self, digit, x):
        """Digit (int): The desired digit (0-9)"""
        """X (np.array) the training features"""
        """Returns """
        mean = np.array(self.X_mean[digit]).reshape(1, 256)
        if self.use_unit_variance:
            log_prob_x_given_digit = -(256 / 2) * (np.log(2 * np.pi)) - 0.5 * np.dot(np.dot((x - mean), self.X_var), (
                        x.reshape(256, 1) - mean.reshape(256, 1)))
            return log_prob_x_given_digit[0][0]
        else:
            var = np.diag(self.X_var[digit])
            smooth = 10 ** - 9
            prob_constant = (-256 / 2) * np.log(1 / (np.linalg.det(var) * 2 * np.pi + 10 ** - 6))
            # print(var)
            for i in range(256):
                var[i, i] += smooth
        inverse_cov = np.diag([1 / var[i, i] for i in range(256)])
        # log_prob_x_given_digit = -(256 / 2) * (np.log(2 * np.pi) + np.sum(np.log([var[i, i] for i in range(256)]))) - 0.5 * np.dot(np.dot((x - mean), inverse_cov), (x.reshape(256, 1) - mean.reshape(256, 1)))
        log_prob_x_given_digit = prob_constant - 0.5 * np.dot(np.dot((x - mean), inverse_cov), (x.reshape(256, 1) - mean.reshape(256, 1)))

        return log_prob_x_given_digit[0][0]


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        predictions = []
        for x in X:
            posterior_probs = []
            for digit in self.priors:
                prior_prob = self.priors[digit]
                log_cond_prob = self.gaussian_prob(digit, x.reshape(1, 256))
                posterior_probs.append(np.log(prior_prob) + log_cond_prob)
            # print(posterior_probs)
            predictions.append(posterior_probs.index(max(posterior_probs)))
        # print(predictions)
        return np.array(predictions)



    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        predictions = self.predict(X)
        acc = np.sum(predictions == y) / len(y)
        print(acc)
        return acc

# g = CustomNBClassifier(use_unit_variance=False)
# g.fit(train_features, train_labels)
# g.gaussian_prob(1, train_features[0].reshape(1, 256))
# g.score(test_features, test_labels)




'''For step 17 we will compare the performance (score) of the following four sklearn estimators: Gaussian Naive Bayes,
 Nearest Neighbors, SVM with linear kernel and SVM with RBF kernel'''

def compare_classifiers(X, y):
    clf1 = GaussianNB()
    clf2 = KNeighborsClassifier()
    clf3 = SVC(kernel='linear')
    clf4 = SVC(kernel='rbf')
    clf1.fit(X, y)
    print('Sklearn Gaussian Naive Bayes estimator gives a score of:', 100 * clf1.score(test_features, test_labels), '%')
    clf2.fit(X, y)
    print('Sklearn K-Neighbors estimator gives a score of:', 100 * clf2.score(test_features, test_labels), '%')
    clf3.fit(X, y)
    print('Sklearn SVM with linear kernel estimator gives a score of:', 100 * clf3.score(test_features, test_labels), '%')
    clf4.fit(X, y)
    print('Sklearn SVM with rbf kernel estimator gives a score of:', 100 * clf4.score(test_features, test_labels), '%')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cm1 = confusion_matrix(test_labels, clf1.predict(test_features))
    sns.heatmap(cm1, annot=True, fmt='d', ax=ax1)
    ax1.set_title('Naive Bayes')
    cm2 = confusion_matrix(test_labels, clf2.predict(test_features))
    sns.heatmap(cm2, annot=True, fmt='d', ax=ax2)
    ax2.set_title('K-Neighbors')
    fig, (ax3, ax4) = plt.subplots(1, 2)
    cm3 = confusion_matrix(test_labels, clf3.predict(test_features))
    sns.heatmap(cm3, annot=True, fmt='d', ax=ax3)
    ax3.set_title('Linear SVM')
    cm4 = confusion_matrix(test_labels, clf4.predict(test_features))
    sns.heatmap(cm4, annot=True, fmt='d', ax=ax4)
    ax4.set_title('RBF SVM')
    plt.show()


# compare_classifiers(train_features, train_labels)



'''Step 18 is to implement a Voting Classifier and a Bagging Classifier in order to be able to choose the best
 classifier or the best combination of classifiers.'''

'''Firstly we implement the Voting Method'''
def evaluate_voting_classifier(X, y, folds=5):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf1 = GaussianNB()
    clf2 = SVC(kernel='rbf')
    clf3 = SVC(kernel='linear')
    clf4 = KNeighborsClassifier()
    clf5 = BaggingClassifier()
    clf = VotingClassifier(estimators=[('GNB', clf1), ('L-SVC', clf2), ('RBF-SVC', clf3),
                                         ('K-NN', clf4), ('BAG', clf5)], voting='hard')
    clf.fit(X, y)
    print('Sklearn Voting Classifier gives an accuracy of:', 100 * clf.score(test_features, test_labels),
          '%')
    cross_val = cross_val_score(clf, X, y, cv=KFold(n_splits=folds))
    print("The 5-fold cross-validation score of the Sklearn Voting estimator"
          " is %f +-%f" % (100 * np.mean(cross_val), 100 * np.std(cross_val)))
    return cross_val

# evaluate_voting_classifier(train_features, train_labels)


def evaluate_bagging_classifier(X, y, folds=5):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = BaggingClassifier()
    clf.fit(X, y)
    print('Sklearn Bagging Classifier gives an accuracy of:', 100 * clf.score(test_features, test_labels),
          '%')
    cross_val = cross_val_score(clf, X, y, cv=KFold(n_splits=folds))
    print("The 5-fold cross-validation score of the Sklearn Bagging estimator"
          " is %f +-%f" % (100 * np.mean(cross_val), 100 * np.std(cross_val)))
    return cross_val

# evaluate_bagging_classifier(train_features, train_labels)










'''Step 19 is to construct a Pytorch Neural Network that will do the classification task'''

class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        """In the init method we initialize the model (the architecture of the network), the loss function and the
        optimizer."""
        self.model = nn.Sequential(nn.Linear(256, 200), nn.Sigmoid(), nn.Linear(200, 10))
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def fit(self, X, y):
        """X (np.array): training features
        y (np.array): training labels"""

        """The fit method feeds the data into the neural network. In this case we do not partition the data into batches,
        since the training data is very small and the problem simple enough not to slow down the computations. The
        function returns the now trained model."""
        data = np.insert(X, 0, y, axis=1)
        np.random.shuffle(data)
        train_loader = T.tensor(data[:, 1:], dtype=T.float32)
        val_loader = T.LongTensor(data[:, 0])
        self.model.train()
        epochs = 100
        for _ in range(epochs):
            self.optimizer.zero_grad()
            pred = self.model(train_loader)
            loss = self.criterion(pred, val_loader)
            loss.backward()
            self.optimizer.step()
        return self


    def predict(self, X):
        """X : test data (np.array)"""
        """We turn the model into evaluation mode and we feed the test data into it. The function return the array
        of predictions (np.array)"""
        self.model.eval()
        predictions = []
        with T.no_grad():
            for x in X:
                x = T.tensor(x, dtype=T.float32)
                out = self.model(x).detach().numpy()
                pred = max(out)
                predictions.append(np.where(out == pred)[0])
        # print(predictions)
        return np.array(predictions)

    def score(self, X, y):
        """We turn the model into evaluation mode and we get the predictions from the predict method. We then simply
        measure how many of these predictions match with the test labels and keep the score. The method return a float."""
        self.model.eval()
        preds = self.predict(X)
        acc = 0
        for i in range(len(y)):
            if preds[i][0] == y[i]:
                acc += 1
        acc /= len(y)
        # print(100 * acc)
        return acc

# class PytorchNNModel_with_batches(BaseEstimator, ClassifierMixin):
#     def __init__(self):
#         """In the init method we initialize the model (the architecture of the network), the loss function and the
#         optimizer."""
#         self.model = nn.Sequential(nn.Linear(256, 200), nn.ReLU(), nn.Linear(200, 10))
#         self.criterion = CrossEntropyLoss()
#         self.optimizer = Adam(self.model.parameters(), lr=0.005)
#
#     def fit(self, X):
#         """X :training set (features and labels"""
#
#         """The fit method feeds the data into the neural network. In this case we do not partition the data into batches,
#         since the training data is very small and the problem simple enough not to slow down the computations. The
#         function returns the now trained model."""
#
#         np.random.shuffle(X)
#         features = X[:, 1:]
#         labels = X[:, 0]
#         feature_batches = np.array_split(features, 100)
#         label_batches = np.array_split(labels, 100)
#         self.model.train()
#         epochs = 100
#         for _ in range(epochs):
#             for i in range(100):
#                 self.optimizer.zero_grad()
#                 # print(feature_batches.shape)
#                 pred = self.model(T.tensor(feature_batches[i], dtype=T.float32))
#                 loss = self.criterion(pred, T.LongTensor(label_batches[i]))
#                 loss.backward()
#                 self.optimizer.step()
#         return self
#
#
#     def predict(self, X):
#         """X : test data (np.array)"""
#         """We turn the model into evaluation mode and we feed the test data into it. The function return the array
#         of predictions (np.array)"""
#         self.model.eval()
#         predictions = []
#         with T.no_grad():
#             for x in X:
#                 x = T.tensor(x, dtype=T.float32)
#                 out = self.model(x).detach().numpy()
#                 pred = max(out)
#                 predictions.append(np.where(out == pred)[0])
#         # print(predictions)
#         return predictions
#
#     def score(self, X, y):
#         """We turn the model into evaluation mode and we get the predictions from the predict method. We then simply
#         measure how many of these predictions match with the test labels and keep the score. The method return a float."""
#         self.model.eval()
#         preds = self.predict(X)
#         acc = 0
#         for i in range(len(y)):
#             if preds[i][0] == y[i]:
#                 acc += 1
#         acc /= len(y)
#         print(100 * acc)
#         return acc


# net = PytorchNNModel()
# net.fit(X_training)
# # net.predict(test_features)
# net.score(test_features, test_labels)


# net = PytorchNNModel_with_batches()
# net.fit(X_training)
# # net.predict(test_features)
# net.score(test_features, test_labels)


def evaluate_nn_classifier(X, y, folds=5):
    """ Create a pytorch nn classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = PytorchNNModel()
    # cross_val = cross_val_score(clf, X, y, cv=KFold(n_splits=folds), scoring="accuracy")
    # clf.fit(train_features, train_labels)
    # print('Neural Network estimator gives an accuracy of:', 100 * clf.score(test_features, test_labels),
    #       '%')
    # print("The 5-fold cross-validation score of the Neural Network classifier"
    #       " is %f +-%f" % (100 * np.mean(cross_val), 100 * np.std(cross_val)))
    plot_learning_curve(clf, X, y)



evaluate_nn_classifier(train_features, train_labels)
