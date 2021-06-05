# Import matplotlib library
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from time import time
import logging

# Import scikit-learn library
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
from statistics import mean
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from time import time


#  Data Preprocessing, this is mainly for comparison purposes, we have also declared similar values below for the actual classification
def processing( min_faces, fraction = 1, verbose = True, test_split = True, special_params = False):

  if special_params:
      dataset = fetch_lfw_people(min_faces_per_person = min_faces,  slice_=None,  resize=1.)
  else:
      dataset = fetch_lfw_people(min_faces_per_person = min_faces)
  X = dataset.data
  n_features = X.shape[1]
  
  if verbose: 
    targets = dataset.target_names
    n_classes = targets.shape[0]
    
  y = dataset.target

  # select fraction from dataset
  if fraction != 1:
    X, n, y, n = train_test_split(X, y, train_size = fraction, random_state = 42)
  
  X_scaled = MinMaxScaler().fit_transform(X)

  if test_split:
    train_data, test_data, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42) 
    return train_data, y_train, test_data, y_test
  return X_scaled, y

# used for better plots
plt.style.use('ggplot')
# download the LFW_people's dataset
lfw_people = fetch_lfw_people(min_faces_per_person = 70, resize = 0.4)
  
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
  
# Instead of providing 2D data, X has data already in the form  of a vector that
# is required in this approach.
X = lfw_people.data
n_features = X.shape[1]
  
# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
  
# Print Details about dataset
print("Number of Data Samples: % d" % n_samples)
print("Size of a data sample: % d" % n_features)
print("Number of Class Labels: % d" % n_classes)

# Finding Optimum Number of Principle Component, this is helpful to set the right parameters 
pca = PCA()
pca.fit(X)
plt.figure(1, figsize=(12,8))
plt.plot(pca.explained_variance_, linewidth=2)
plt.xlabel('Components')
plt.ylabel('Explained Variaces')
plt.show()

# Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
g_precision, g_recall, g_f_score, _ = [mean(x) for x in precision_recall_fscore_support(y_test, y_pred)]
print("Precision : %s, Recall : %s, F-score : %s" %( g_precision, g_recall, g_f_score))

# Comparison between SVM methods to determine best for usage in classification
min_faces = 70

X_scaled, y = processing(min_faces, verbose = False, test_split=False)

pca = PCA(n_components = 150)
X_reduced = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_reduced, y, test_size = 0.2, random_state = 42)

#  parameters for comparisson 
p_grid = {"C": [1, 10, 100, 1000],
          "gamma": [0.0001, 0.001, .01, .1]}

polynomial_svm = SVC(kernel = 'poly')
rbf_svm = SVC(kernel = 'rbf')

N_TRIALS = 3

polynomial_scores = np.zeros(N_TRIALS)
rbf_scores = np.zeros(N_TRIALS)

polynomial_clf = GridSearchCV(estimator = polynomial_svm, param_grid=p_grid)
rbf_clf = GridSearchCV(estimator = rbf_svm, param_grid=p_grid)
polynomial_clf.fit(X_train_pca, y_train)
rbf_clf.fit(X_train_pca, y_train)
print("Best parameters for polynomial kernel : ", polynomial_clf.best_estimator_)
print("Best parameters for rbf kernel : ", rbf_clf.best_estimator_)

# Performing training
polynomial_svm = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='poly')
rbf_svm = SVC(C=100, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf')
    
polynomial_y_pred = polynomial_svm.fit(X_train_pca, y_train).predict(X_test_pca)
rbf_y_pred = rbf_svm.fit(X_train_pca, y_train).predict(X_test_pca)

polynomial_precision, polynomial_recall, polynomial_f_score, _ = [mean(x) for x in precision_recall_fscore_support(y_test, polynomial_y_pred)]
rbf_precision, rbf_recall, rbf_f_score, _ =  [mean(x) for x in precision_recall_fscore_support(y_test, rbf_y_pred)]

polynomial_svm = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='poly')

rbf_svm = SVC(C=100, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf')

polynomial_y_pred = polynomial_svm.fit(X_train_pca, y_train).predict(X_test_pca)
rbf_y_pred = rbf_svm.fit(X_train_pca, y_train).predict(X_test_pca)

polynomial_precision, polynomial_recall, polynomial_f_score, _ = [mean(x) for x in precision_recall_fscore_support(y_test, polynomial_y_pred)]
rbf_precision, rbf_recall, rbf_f_score, _ =  [mean(x) for x in precision_recall_fscore_support(y_test, rbf_y_pred)]

fig, axs = plt.subplots( figsize=(10, 7))
fig.suptitle('Comparison of SVM metrics')

x = [0, 1, 2]
width = 0.15

# Precision
gaussian = axs.bar([a - width for a in x], [g_precision, g_recall, g_f_score], width, label='Naive Bayes')
polynomial = axs.bar([a for a in x], [polynomial_precision, polynomial_recall, polynomial_f_score], width, label='Polynomial SVM')
rbf= axs.bar([a + width for a in x], [rbf_precision, rbf_recall, rbf_f_score], width, label='RBF SVM')

axs.set_ylabel('Scores')
axs.set_xticks(x)
axs.set_xticklabels(["Precision", "Recall", "F score"])
axs.legend()

plt.show()

# Visualization
def plot_gallery(images, titles, h, w, rows=3, cols=4):
    plt.figure()
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())

def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 42)
print("size of training Data is % d and Testing Data is % d" %(
        y_train.shape[0], y_test.shape[0]))

n_components = 150
  

pca = PCA(n_components = n_components, svd_solver ='randomized',
          whiten = True).fit(X_train)

t0 = time()
eigenfaces = pca.components_.reshape((n_components, h, w))

# Ploting the variance will show that is not necessary to consider a large number of dimensions in order to build a face recognition model
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Components')
plt.ylabel('Cumulative Variance');

print("Projecting the input data on the eigenfaces orthonormal basis")

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Sample Data point after applying PCA\n", X_train_pca[0])
print("-----------------------------------------------------")
print("Dimesnsions of training set = % s and Test Set = % s"%(
        X_train.shape, X_test.shape))
        
print("Fitting the classifier to the training set")

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel ='rbf', class_weight ='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))

print("Best estimator found by grid search:")
print(clf.best_estimator_)
  
print("Predicting people's names on the test set")

y_pred = clf.predict(X_test_pca)
print("Accuracy score:{:.2f}%".format(metrics.accuracy_score(y_test, y_pred)*100))

plt.figure(2, figsize=(16, 9))
sns.heatmap(metrics.confusion_matrix(y_test, y_pred))

# print classifiction results
print(classification_report(y_test, y_pred, target_names = target_names))
# print confusion matrix
print("Confusion Matrix is:")
print(confusion_matrix(y_test, y_pred, labels = range(n_classes)))

# plot confusion matrix
plt.rcParams.update({'font.size': 10})
plt.rcParams['figure.figsize'] = [10, 10]
plot_confusion_matrix(clf, X_test_pca, y_test, cmap=plt.cm.Blues, display_labels=range(n_classes))

prediction_titles = list(titles(y_pred, y_test, target_names))
plot_gallery(X_test, prediction_titles, h, w)

eigenface_titles = ["eigenface {0}".format(i) for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

"""
References

https://www.geeksforgeeks.org/ml-face-recognition-using-pca-implementation/

https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html

https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/

https://github.com/carloscarvajal1cc/Face-Recognition-with-PCA/blob/master/FaceRecognition.ipynb

https://github.com/essanhaji/face_recognition_pca/blob/master/main.ipynb

https://github.com/maurinl26/LFW_Face_Recognition/blob/master/Face_Recognition_Projet_ML.ipynb

"""

